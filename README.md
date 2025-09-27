# Synthetic to Street: Generative AI-Powered Object Detection

**Research Implementation for AIPR 2025**  
**Contributors:** John Munoz MSDS, Tom Bukowski MSDS, Dr. Mike Busch PhD

---

## 🎯 Executive Summary

This repository contains the implementation of an Motorcycle Compliance Object Detection Model. <br>
We developed an automated compliance monitoring system capable of real-time detection of safety violations. <br>
We utilized real-world data as well as synthetic video data generated with Google’s Veo 2 and Veo 3 models  <br>


### 📊 Key Achievement
- **81% mAP50** accuracy achieved through hybrid synthetic-real data approach
- **86% precision** in detecting safety compliance violations
- **Real-time processing** capability for traffic monitoring systems

---

[![Model Results](example_gifs/model_results.gif)](example_gifs/model_results.gif)
# 🚨 Problem Statement

In April of 2025, Peru passed a new law requiring all motorcycle operators must wear certified helmets and reflective vests displaying license plate numbers.
Manual enforcement presents significant challenges:

- Resource-intensive manual inspections
- Limited scalability for nationwide coverage
- Inconsistent enforcement across regions
- Delayed response to violations

### 💡 Our Solution
An automated detection system utilizing synthetic data generation to overcome data scarcity challenges while maintaining high accuracy in real-world deployments.

[![Model Results_two](example_gifs/dash_exseven.gif)](example_gifs/dash_exseven.gif)

---

## 🔬 Methodology

### Data Generation Pipeline

We developed three progressive models, each building upon lessons learned from previous iterations:
