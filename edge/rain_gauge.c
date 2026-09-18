/*
 * FreeRTOS POSIX/Linux Rain Gauge Virtual Sensor (Phase 1 / Edge Telemetry).
 *
 * Implements a virtual tipping-bucket sensor task emitting JSON payloads to:
 * MQTT topic: sensors/rain/{station_id} on broker localhost:1883
 * Payload format:
 * {"station_id": "WARANGAL_01", "intensity_mm_h": 45.0, "cumulative_6h_mm": 52.4}
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define DEFAULT_STATION_ID "WARANGAL_01"
#define BUCKET_TIP_MM 0.2
#define MQTT_BROKER_IP "127.0.0.1"
#define MQTT_BROKER_PORT 1883
#define PUBLISH_INTERVAL_SEC 10

typedef struct {
    char station_id[64];
    double intensity_mm_h;
    double cumulative_6h_mm;
    unsigned long tip_count;
    time_t last_tip_time;
} RainGaugeState;

static RainGaugeState g_state;
static pthread_mutex_t g_state_mutex = PTHREAD_MUTEX_INITIALIZER;
static volatile int g_running = 1;

/* FreeRTOS POSIX Tipping-bucket simulation task */
void* vRainGaugeTipTask(void* pvParameters) {
    (void)pvParameters;
    printf("[FreeRTOS/POSIX] Tipping-Bucket contact closure task initialized for %s.\n", g_state.station_id);

    while (g_running) {
        // Convective storm event simulation: tip every 1.5 to 3.0 seconds
        usleep((1500 + (rand() % 1500)) * 1000);

        pthread_mutex_lock(&g_state_mutex);
        g_state.tip_count++;
        g_state.cumulative_6h_mm += BUCKET_TIP_MM;
        g_state.last_tip_time = time(NULL);

        // Instantaneous intensity: 0.2mm per 2s average -> ~36-48 mm/h calibrated storm intensity
        g_state.intensity_mm_h = 45.0 + ((rand() % 100) - 50) * 0.1;
        pthread_mutex_unlock(&g_state_mutex);
    }
    return NULL;
}

/* MQTT 3.1.1 Publisher */
int mqtt_publish_simple(const char* topic, const char* json_payload) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return -1;

    struct sockaddr_in server;
    memset(&server, 0, sizeof(server));
    server.sin_family = AF_INET;
    server.sin_port = htons(MQTT_BROKER_PORT);
    inet_pton(AF_INET, MQTT_BROKER_IP, &server.sin_addr);

    struct timeval tv = {.tv_sec = 2, .tv_usec = 0};
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    if (connect(sock, (struct sockaddr*)&server, sizeof(server)) < 0) {
        close(sock);
        return -1;
    }

    // MQTT CONNECT Packet
    uint8_t connect_packet[] = {
        0x10, 0x14,                                     // CONNECT, len 20
        0x00, 0x04, 'M', 'Q', 'T', 'T',                 // Protocol Name
        0x04,                                           // Level 3.1.1
        0x02,                                           // Clean Session
        0x00, 0x3C,                                     // Keep Alive 60s
        0x00, 0x08, 'E', 'D', 'G', 'E', '_', 'R', 'A', 'I' // Client ID
    };
    if (write(sock, connect_packet, sizeof(connect_packet)) < 0) {
        close(sock);
        return -1;
    }

    // Read CONNACK
    uint8_t connack[4];
    if (read(sock, connack, sizeof(connack)) <= 0 || connack[3] != 0x00) {
        close(sock);
        return -1;
    }

    // MQTT PUBLISH Packet (QoS 0)
    uint16_t topic_len = strlen(topic);
    uint16_t payload_len = strlen(json_payload);
    uint16_t rem_len = 2 + topic_len + payload_len;

    uint8_t pub_header[5];
    pub_header[0] = 0x30;
    pub_header[1] = rem_len;
    pub_header[2] = (topic_len >> 8) & 0xFF;
    pub_header[3] = topic_len & 0xFF;

    write(sock, pub_header, 4);
    write(sock, topic, topic_len);
    write(sock, json_payload, payload_len);

    // MQTT DISCONNECT
    uint8_t disc[] = {0xE0, 0x00};
    write(sock, disc, 2);

    close(sock);
    return 0;
}

/* MQTT Telemetry broadcast task */
void* vMqttBroadcastTask(void* pvParameters) {
    (void)pvParameters;
    char payload_buf[256];
    char topic_buf[128];
    snprintf(topic_buf, sizeof(topic_buf), "sensors/rain/%s", g_state.station_id);

    printf("[FreeRTOS/POSIX] Emitting telemetry to MQTT topic: %s on localhost:1883\n", topic_buf);

    while (g_running) {
        sleep(PUBLISH_INTERVAL_SEC);

        pthread_mutex_lock(&g_state_mutex);
        snprintf(payload_buf, sizeof(payload_buf),
            "{\"station_id\": \"%s\", \"intensity_mm_h\": %.1f, \"cumulative_6h_mm\": %.1f}",
            g_state.station_id, g_state.intensity_mm_h, g_state.cumulative_6h_mm
        );
        pthread_mutex_unlock(&g_state_mutex);

        int rc = mqtt_publish_simple(topic_buf, payload_buf);
        if (rc == 0) {
            printf("[FreeRTOS/POSIX] Published telemetry: %s\n", payload_buf);
        } else {
            printf("[FreeRTOS/POSIX] Telemetry buffer (broker offline): %s\n", payload_buf);
        }
    }
    return NULL;
}

int main(int argc, char* argv[]) {
    printf("=== FreeRTOS POSIX Rain Gauge Sensor Initializing ===\n");
    memset(&g_state, 0, sizeof(g_state));
    strncpy(g_state.station_id, (argc > 1) ? argv[1] : DEFAULT_STATION_ID, sizeof(g_state.station_id) - 1);
    g_state.intensity_mm_h = 45.0;
    g_state.cumulative_6h_mm = 52.4;

    pthread_t tip_tid, mqtt_tid;
    pthread_create(&tip_tid, NULL, vRainGaugeTipTask, NULL);
    pthread_create(&mqtt_tid, NULL, vMqttBroadcastTask, NULL);

    int runtime_sec = (argc > 2) ? atoi(argv[2]) : 0;
    if (runtime_sec > 0) {
        printf("[FreeRTOS/POSIX] Running for %d seconds...\n", runtime_sec);
        sleep(runtime_sec);
        g_running = 0;
        pthread_join(tip_tid, NULL);
        pthread_join(mqtt_tid, NULL);
        printf("[FreeRTOS/POSIX] Rain gauge terminated.\n");
    } else {
        pthread_join(tip_tid, NULL);
        pthread_join(mqtt_tid, NULL);
    }

    return 0;
}
