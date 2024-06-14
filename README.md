# Traffic Flow Analysis System at Intersections Using Video Footage

## Overview

This project is the culmination of a research thesis focused on the development of a system to calculate and analyze traffic flow parameters at intersections using video camera footage. The project addresses the growing issue of effective traffic management at intersections in the face of increasing vehicle numbers and road infrastructure load.

## Motivation

Accurate and timely data collection on traffic movement is a critical component in improving traffic management at intersections. The goal of this project is to explore modern methods of collecting traffic flow data using video streams and develop a system for their analysis.

## Key Features

- **Object Detection and Tracking**: Utilizes state-of-the-art object detection and tracking algorithms, including neural network-based algorithms such as Faster R-CNN, YOLO, and tracking algorithms like SORT, DeepSort, ByteTrack.
- **Traffic Flow Analysis**: Analyzes the parameters of traffic flow, including the number and type of vehicles, number of pedestrians, and their direction of movement.
- **Data Visualization and Statistics**: Visualizes the data and statistics collected, providing clear and actionable insights.
- **Data Storage**: Saves collected data on a server for further analysis and reporting.

## System Design

The designed system processes video stream data to:
- Detect and track vehicles and pedestrians.
- Count the number of vehicles and pedestrians and categorize them by type.
- Analyze the direction of movement.
- Store the processed data on a server.
- Display the analyzed data in a user-friendly interface.

## Installation

1. Clone the repository:
    ```sh
    git clone git@github.com:samb1232/IntersectionTracking.git
    cd IntersectionTracking
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the server:
    ```sh
    python main.py
    ```
   
    Or run multiprocessing version:
    ```sh
    python main_optimized.py
    ```

## Usage

1. Place video files in `test_videos` directory.
2. Run the script
3. Change system settings in file `configs/app_cinfig.yaml` if you want. 
4. If you want to use database and additional services, run docker and execute file `run_services.bat`

 
## Appreciations

This project based on repository: https://github.com/Koldim2001/TrafficAnalyzer

Many thanks from me to user Koldim2001. See his work too.