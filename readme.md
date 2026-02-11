# P2CODE Service and Infrastructure Telemetry

**WIP** This repository is currently incomplete and being updated

## Overview 
The Telemetry component within P2CODE is charged with telemetry acquisition from the two application testbeds, long term storage, as well as analysis to extract insights with regard to performance as well as defect source attribution. (Root Cause Analysis)

This repository provides the following images:

1. **Otel_stack** OpenTelemetry based stack handling log, metric, and trace acquisition via the OpenTelemetry Collector, Loki, ElasticSearch, Jaeger, Prometheus. This stack was unused in the actual P2CODE deployment, efforts were limited to metric collection.
2. **Mimir** Mimir based stack, with data coming in from Testbed Prometheus instances via remote write.
3. **Insights** Machine learning tooling for insight extraction, based on Python. Uses prometheus_api_client to query Mimir data, as well as prometheus_remote_writer to publish extracted insights.
4. **tcdf** Customization of the Temporal Causal Discovery Framework, deep learning toolkit for causal relation discovery.
5. **MeDIL** Measuring Dependence Inducing Latent - Application of open source MeDIL to P2CODE telemetry.
6. **Others**

## Architecture

- **ALB — Entrypoint to P2CODE AWS Testbed**  

- **EC2 #1 — Mimir - Grafana - WireGuard Zero Trust Connector**  
  Runs Mimir, the analytics pipelines, as well as Grafana. Instance is connected to the ZTC Fabric to permit telemetry push from the P2CODE testbeds.

## Usage
