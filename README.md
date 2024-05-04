# SPAPT
Signal Processing for Analog Phaser Tomography.

This project comprises tomography software featuring a graphical user interface (GUI) and a simulator designed for the Analog Devices CN0566 Hybrid Phase Kit.

## Project Overview
Our project aims to pioneer the utilization of cost-effective one-dimensional analog phase array technology in radar tomography. By integrating the cn0566 phaser array onto a rotational platform, aligned axis with center of phaser, we aim to pioneer three-dimensional tomography synthesis through inventive signal processing algorithms.
![imaging](/image/imaging.gif)
![sar](/image/sar.gif)
## Project Objects

__Hardware Implementation:__
Integrate the cn0566 phaser array onto a rotational platform. Ensure meticulous alignment of the phaser with the rotational axis to enhance data collection accuracy.

__Signal Processing Algorithm:__
Optimize signal integration by reducing buffering for enhanced efficiency. Consider array and antenna patterns in refining the signal processing algorithm for improved tomography synthesis.

__Simulation Environment and User Interface:__
Augment the existing simulator with additional products and command handles.
Transition the state machine to the signal processing unit, relieving the phaser of heavy processing loads.
Implement an intuitive user interface with command hosts for dynamic adjustments in radar parameters and real-time display options.


## Installation
SPAPT phaser GUI needs [pyadi-iio](https://github.com/analogdevicesinc/pyadi-iio), [libiio](https://github.com/analogdevicesinc/libiio), [pyqtgraph](https://github.com/pyqtgraph/pyqtgraph) and [PyQt5](https://pypi.org/project/PyQt5/)

>[!CAUTION]
>This project utilizes the TDD engine feature of pyadi-iio and requires the use of the `origin/cn0566_dev_phaser_merge` branch or its equivalent in future updates.

SPAPT simulator needs [numpy](https://pypi.org/project/numpy/), [matplotlib](https://pypi.org/project/matplotlib/) and [scipy](https://pypi.org/project/scipy/).

Simply clone SPAPT form repository.

```bash
git clone https://github.com/TMinRay/SPAPT.git
```

## Usage
To start the phaser GUI
```bash
$ cd spapt
$ python phaser_qt_imaging_radar.py
```
![UI](/image/UI.png)

To start the simulator
```bash
$ cd simulator
$ python simradar.py
```
See the [simulator/README.md](simulator/README.md) for example and command table.

## Conclusion
Our project has the potential to demonstrate the feasibility of utilizing a one-dimensional phaser array configuration for acquiring three-dimensional tomography. This innovative approach could find applications in space-borne radar systems. The technique presents a significant advantage in terms of weight savings—fewer elements translate to lighter payloads, resulting in substantial budget savings. Remarkably, this is achieved without compromising data quality, making our solution a compelling prospect for cost-effective yet high-quality radar tomography applications, especially in space-based scenarios.

