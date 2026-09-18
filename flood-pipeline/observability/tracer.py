"""
OpenTelemetry Instrumentation and ADOT Exporter Setup (Phase 7).

Configures OpenTelemetry SDK with OTLP Exporter targeting AWS Distro for OpenTelemetry (ADOT) Collector.
Provides resilient no-op fallback when running unit tests outside of Docker.
"""

import os
import socket
import logging
from urllib.parse import urlparse
from contextlib import contextmanager
from typing import Optional, Generator

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger("FloodObservability")

_INITIALIZED = False
_TRACER: Optional[trace.Tracer] = None


class NoOpSpanExporter(SpanExporter):
    """Silent no-op span exporter for local testing without backing Docker collector."""
    def export(self, spans):
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


def _is_endpoint_reachable(host: str, port: int, timeout_sec: float = 0.2) -> bool:
    """Quick socket probe to check if collector endpoint is listening."""
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True
    except Exception:
        return False


def setup_telemetry(
    service_name: str = "flood-inundation-pipeline",
    grpc_endpoint: str = "localhost:4317",
    http_endpoint: str = "http://localhost:4318/v1/traces",
    prefer_grpc: bool = False
) -> trace.Tracer:
    """
    Initializes OTel TracerProvider and adds OTLP SpanProcessor for ADOT collector.
    Gracefully falls back to NoOpSpanExporter when running unit tests outside Docker.
    """
    global _INITIALIZED, _TRACER
    if _INITIALIZED and _TRACER is not None:
        return _TRACER

    resource = Resource.create({
        "service.name": service_name,
        "service.namespace": "hydrology.alerting",
        "deployment.environment": "local-dev"
    })

    provider = TracerProvider(resource=resource)
    exporter = None
    exporter_type = "none"

    if prefer_grpc:
        try:
            parts = grpc_endpoint.split(":")
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 4317
            if _is_endpoint_reachable(host, port):
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GRPCSpanExporter
                exporter = GRPCSpanExporter(endpoint=grpc_endpoint, insecure=True)
                exporter_type = f"OTLP/gRPC ({grpc_endpoint})"
            else:
                logger.debug("gRPC endpoint %s not reachable, probing HTTP...", grpc_endpoint)
        except Exception as ex:
            logger.debug("gRPC exporter initialization skipped: %s", ex)

    if exporter is None:
        try:
            parsed = urlparse(http_endpoint)
            host = parsed.hostname or "localhost"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            if _is_endpoint_reachable(host, port):
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HTTPSpanExporter
                exporter = HTTPSpanExporter(endpoint=http_endpoint, timeout=2)
                exporter_type = f"OTLP/HTTP ({http_endpoint})"
            else:
                logger.debug("HTTP endpoint %s not reachable, using NoOpSpanExporter fallback.", http_endpoint)
        except Exception as ex:
            logger.debug("HTTP exporter initialization skipped: %s", ex)

    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        logger.info("Initialized OpenTelemetry with %s targeting ADOT Collector.", exporter_type)
    else:
        # Resilient no-op fallback when ADOT collector is offline
        provider.add_span_processor(SimpleSpanProcessor(NoOpSpanExporter()))
        logger.info("Initialized OpenTelemetry with NoOpSpanExporter (ADOT collector offline / local test mode).")

    trace.set_tracer_provider(provider)
    _TRACER = trace.get_tracer("flood.pipeline.tracer")
    _INITIALIZED = True
    return _TRACER


def get_tracer() -> trace.Tracer:
    """Returns the globally configured tracer instance."""
    global _TRACER
    if _TRACER is None:
        return setup_telemetry()
    return _TRACER


@contextmanager
def trace_span(name: str, attributes: Optional[dict] = None) -> Generator[trace.Span, None, None]:
    """Context manager for tracing an individual execution stage with attributes."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        yield span
