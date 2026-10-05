# SIH26055 – Smart Scan Strategy for Electronic Warfare

## AI-Based Adaptive Spectrum Scan Scheduler

A simulation-based machine learning framework for intelligent scheduling of a limited-bandwidth receiver across a wide frequency spectrum.

**Smart India Hackathon 2026**  
**Problem Statement ID:** SIH26055  
**Problem Statement:** Smart Scan Strategy for Electronic Warfare  
**Theme:** Robotics and Drones  
**Category:** Software  
**Team:** VisionX

---

## 📌 Problem Statement

Wideband spectrum surveillance requires a receiver to monitor a broad frequency range even though its instantaneous bandwidth is significantly smaller than the overall surveillance bandwidth.

Conventional open-loop scanning strategies generally follow a predetermined frequency sequence. Such strategies may spend valuable scan time on inactive or low-priority frequency bands and increase the delay in detecting newly active emitters.

This project develops a software-based adaptive receiver scheduler that learns from previous observations and intelligently selects the next frequency band to scan.

---

## 🎯 Project Objective

The objective is to develop a machine-learning-based Electronic Support receiver scheduler that:

- Adaptively selects frequency bands for scanning.
- Learns from previous detection hits and misses.
- Models different simulated emitter behaviors.
- Predicts useful frequency and temporal activity.
- Reduces unnecessary scans.
- Improves simulated interception performance.
- Provides quantitative performance evaluation.

---

## 🧠 System Architecture

```text
                 SIMULATED RF ENVIRONMENT
                           │
                           ▼
                  GROUND TRUTH SPECTRUM
                           │
                           ▼
                    VIRTUAL RECEIVER
                           │
                    ┌──────┴──────┐
                    │             │
                   HIT           MISS
                    │             │
                    └──────┬──────┘
                           ▼
                  OBSERVATION MEMORY
                           │
                           ▼
                    ML PREDICTION
                           │
                           ▼
                  SMART SCHEDULER
                           │
                           ▼
                    NEXT BAND
                           │
                           └──────────► Repeat
