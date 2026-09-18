"""
OpenTelemetry Instrumentation and ADOT Exporter Setup (Phase 7).

Configures OpenTelemetry SDK with OTLP Exporter targeting AWS Distro for OpenTelemetry (ADOT) Collector.
"""

import os
import logging
from contextlib import contextmanager
from typing import Optional, Generator

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger("FloodObservability")

_INITIALIZED = False
_TRACER: Optional[trace.Tracer] = None


def setup_telemetry(
    service_name: str = "flood-inundation-pipeline",
    grpc_endpoint: str = "localhost:4317",
    http_endpoint: str = "http://localhost:4318/v1/traces",
    prefer_grpc: bool = False
) -> trace.Tracer:
    """
    Initializes OTel TracerProvider and adds OTLP SpanProcessor for ADOT collector.
    Gracefully falls back between gRPC, HTTP, and Console exporters.
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
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GRPCSpanExporter
            exporter = GRPCSpanExporter(endpoint=grpc_endpoint, insecure=True)
            exporter_type = f"OTLP/gRPC ({grpc_endpoint})"
        except Exception as ex:
            logger.debug("gRPC exporter initialization skipped: %s", ex)

    if exporter is None:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HTTPSpanExporter
            exporter = HTTPSpanExporter(endpoint=http_endpoint)
            exporter_type = f"OTLP/HTTP ({http_endpoint})"
        except Exception as ex:
            logger.warning("HTTP exporter initialization failed: %s", ex)

    if exporter is not None:
        # Use SimpleSpanProcessor during testing/local pipeline to immediately push spans without batch delay
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        logger.info("Initialized OpenTelemetry with %s targeting ADOT Collector.", exporter_type)
    else:
        # Fallback to local console
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        logger.info("Initialized OpenTelemetry with Console exporter fallback.")

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
