# Smart Scan Strategy for Electronic Warfare

## 1. Project Overview

The Smart Scan Strategy is a software-based simulation and machine-learning framework designed to improve the efficiency of wideband spectrum surveillance.

A receiver that monitors a wide frequency spectrum generally has a much smaller instantaneous bandwidth than the total spectrum that needs to be observed. Therefore, the receiver must continuously decide **which frequency band to scan and when to scan it**.

A conventional scanning system may follow a fixed sequence regardless of whether a particular frequency band is active or inactive. This can result in unnecessary scan opportunities being spent on inactive regions of the spectrum.

The proposed system introduces an adaptive scan scheduler that learns from previous observations and dynamically prioritizes frequency bands based on their observed activity.

The current implementation provides a simulated RF environment, virtual receiver, multiple scan strategies, machine-learning-based scheduling, performance measurement, visualization and experiment data generation.

---

# 2. Problem Definition

Wideband spectrum monitoring can be considered a two-dimensional search problem involving:

* Frequency
* Time

At every time step, the receiver must select a frequency band for observation.

The environment may contain emitters whose activity changes with time and frequency. Some emitters may be continuous, some periodic, some random, some burst-based, while frequency-agile emitters may change their operating frequency.

The main challenge is:

> How can a receiver intelligently select the next frequency band using information obtained from previous scans while continuing to monitor the wider spectrum?

The objective is to reduce unnecessary scanning and improve the probability and speed of intercepting relevant signal activity.

---

# 3. Objectives

The main objectives of the project are:

1. Simulate a configurable RF environment.
2. Generate transmission activity across multiple frequency bands and time slots.
3. Simulate receiver detection behavior.
4. Record receiver hits and misses.
5. Maintain historical observations for each frequency band.
6. Compare conventional and adaptive scanning strategies.
7. Apply machine learning to predict useful frequency-band activity.
8. Dynamically select the next frequency band.
9. Measure detection and interception performance.
10. Provide visualization and downloadable simulation results.

---

# 4. Concept of Operation

The overall operation of the system is:

```text
                RF Environment
                      |
                      v
              Emitter Activity
                      |
                      v
              Ground Truth Data
                      |
                      v
               Virtual Receiver
                      |
                +-----+-----+
                |           |
               HIT         MISS
                |           |
                +-----+-----+
                      |
                      v
              Observation Memory
                      |
                      v
              Feature Generation
                      |
                      v
             Machine Learning Model
                      |
                      v
              Band Priority Score
                      |
                      v
              Next Band Selection
                      |
                      v
              Receiver Scans Band
                      |
                      +-----------> Repeat
```

The process operates continuously during the simulation.

---

# 5. RF Environment Simulation

The software creates a simulated frequency spectrum divided into multiple frequency bands.

For every time slot, each band can have one of two basic states:

```text
Transmission
No Transmission
```

The environment maintains the actual transmission state as ground-truth information.

For example:

```text
             Time
          T1 T2 T3 T4 T5

Band 1     0  1  1  0  0
Band 2     0  0  1  1  0
Band 3     1  0  0  0  1
Band 4     0  0  0  1  1
```

Where:

```text
1 = transmission
0 = no transmission
```

The receiver only observes the frequency band that it chooses to scan.

---

# 6. Emitter Models

The simulation supports several types of emitter behavior.

## 6.1 Continuous Emitter

A continuous emitter remains active for an extended period or throughout the simulation.

This provides a simple case for evaluating whether the scheduler can repeatedly identify an active frequency.

---

## 6.2 Periodic Emitter

A periodic emitter becomes active according to a recurring temporal pattern.

This allows the scheduler to learn repeated activity patterns.

---

## 6.3 Random Emitter

A random emitter changes its activity according to a stochastic process.

This represents a more difficult prediction scenario because previous observations may have limited predictive value.

---

## 6.4 Burst Emitter

A burst emitter becomes active for short periods and then becomes inactive.

This tests whether the scheduler can respond quickly to temporary signal activity.

---

## 6.5 Frequency-Agile Emitter

A frequency-agile emitter changes its operating frequency over time.

This provides a challenging scenario because activity can move between frequency bands.

---

# 7. Virtual Receiver

The project includes a software-based receiver model.

The receiver does not physically capture RF signals. Instead, it interacts with the simulated RF environment.

The receiver is configured using parameters such as:

* Probability of detection
* Probability of false alarm
* Selected frequency band
* Current time slot

For every scan, the receiver produces an observation.

The basic outcomes are:

```text
HIT
MISS
```

These observations are stored and used by the scheduler.

---

# 8. Probability of Detection

The receiver includes a configurable probability of detection (`Pd`).

When a transmission is actually present in the selected band, `Pd` determines the likelihood that the receiver correctly detects it.

This allows receiver performance to be simulated under different detection conditions.

---

# 9. Probability of False Alarm

The receiver also includes a probability of false alarm (`Pfa`).

When no transmission is present, the receiver may st
