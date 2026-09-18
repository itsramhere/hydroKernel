/*
 * FreeRTOS POSIX Rain Gauge Sensor Simulator (Phase 2).
 *
 * Simulates tipping-bucket rain gauge contact closures, computes:
 * - Instantaneous rainfall rate (mm/hr)
 * - Cumulative rolling accumulation (mm)
 * Formats JSON telemetry and publishes to Mosquitto MQTT broker.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define STATION_ID "STATION_WARANGAL_01"
#define BUCKET_TIP_MM 0.2
#define MQTT_BROKER_IP "127.0.0.1"
#define MQTT_BROKER_PORT 1883
#define PUBLISH_INTERVAL_SEC 10

typedef struct {
    double instant_rate_mm_hr;
    double cumulative_1hr_mm;
    double cumulative_6hr_mm;
    unsigned long tip_count;
    time_t last_tip_time;
} RainGaugeState;

static RainGaugeState g_state = {0};
static pthread_mutex_t g_state_mutex = PTHREAD_MUTEX_INITIALIZER;
static volatile int g_running = 1;

/* Simulates tipping-bucket contact closure task */
void* vRainGaugeTipTask(void* pvParameters) {
    (void)pvParameters;
    printf("[FreeRTOS/POSIX] Tipping-Bucket contact closure task initialized.\n");

    while (g_running) {
        // Heavy convective storm simulation: 1 tip every 1.5 to 3.0 seconds
        usleep((1500 + (rand() % 1500)) * 1000);

        pthread_mutex_lock(&g_state_mutex);
        g_state.tip_count++;
        g_state.cumulative_1hr_mm += BUCKET_TIP_MM;
        g_state.cumulative_6hr_mm += BUCKET_TIP_MM;
        g_state.last_tip_time = time(NULL);

        // Instantaneous rate: tips per hour
        // (0.2 mm / ~2.0 sec) * 3600 sec = ~360 mm/hr burst rate scaled
        g_state.instant_rate_mm_hr = (BUCKET_TIP_MM / 2.0) * 360.0;
        pthread_mutex_unlock(&g_state_mutex);
    }
    return NULL;
}

/* Publishes MQTT 3.1.1 CONNECT and PUBLISH telemetry packet */
int mqtt_publish_simple(const char* topic, const char* json_payload) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return -1;

    struct sockaddr_in server;
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

    // MQTT CONNECT Packet (Clean session, anonymous)
    uint8_t connect_packet[] = {
        0x10, 0x12,                         // Fixed header (CONNECT, length 18)
        0x00, 0x04, 'M', 'Q', 'T', 'T',     // Protocol name
        0x04,                               // Protocol Level (3.1.1)
        0x02,                               // Clean Session
        0x00, 0x3C,                         // Keep Alive 60s
        0x00, 0x06, 'F', 'R', 'E', 'E', 'R', 'T' // Client ID "FREERT"
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

    // MQTT PUBLISH Packet
    uint16_t topic_len = strlen(topic);
    uint16_t payload_len = strlen(json_payload);
    uint16_t rem_len = 2 + topic_len + payload_len;

    uint8_t pub_header[5];
    pub_header[0] = 0x30; // PUBLISH QoS 0
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

/* Telemetry broadcaster task */
void* vMqttBroadcastTask(void* pvParameters) {
    (void)pvParameters;
    char payload_buf[512];
    char topic_buf[128];
    snprintf(topic_buf, sizeof(topic_buf), "sensors/rainfall/%s", STATION_ID);

    printf("[FreeRTOS/POSIX] MQTT Telemetry task active -> %s\n", topic_buf);

    while (g_running) {
        sleep(PUBLISH_INTERVAL_SEC);

        pthread_mutex_lock(&g_state_mutex);
        time_t now = time(NULL);
        snprintf(payload_buf, sizeof(payload_buf),
            "{\"station_id\":\"%s\",\"timestamp\":%ld,\"rate_mm_hr\":%.2f,"
            "\"accum_1hr_mm\":%.2f,\"accum_6hr_mm\":%.2f,\"tips\":%lu}",
            STATION_ID, (long)now, g_state.instant_rate_mm_hr,
            g_state.cumulative_1hr_mm, g_state.cumulative_6hr_mm, g_state.tip_count
        );
        pthread_mutex_unlock(&g_state_mutex);

        int rc = mqtt_publish_simple(topic_buf, payload_buf);
        if (rc == 0) {
            printf("[FreeRTOS/POSIX] Published telemetry: %s\n", payload_buf);
        } else {
            printf("[FreeRTOS/POSIX] Telemetry buffer (MQTT broker offline): %s\n", payload_buf);
        }
    }
    return NULL;
}

int main(int argc, char* argv[]) {
    printf("=== FreeRTOS POSIX Rain Gauge Simulator Starting ===\n");
    pthread_t tip_tid, mqtt_tid;

    int runtime_sec = (argc > 1) ? atoi(argv[1]) : 0;

    pthread_create(&tip_tid, NULL, vRainGaugeTipTask, NULL);
    pthread_create(&mqtt_tid, NULL, vMqttBroadcastTask, NULL);

    if (runtime_sec > 0) {
        printf("[FreeRTOS/POSIX] Running for %d seconds...\n", runtime_sec);
        sleep(runtime_sec);
        g_running = 0;
        pthread_join(tip_tid, NULL);
        pthread_join(mqtt_tid, NULL);
        printf("[FreeRTOS/POSIX] Shutdown cleanly.\n");
    } else {
        pthread_join(tip_tid, NULL);
        pthread_join(mqtt_tid, NULL);
    }

    return 0;
}
