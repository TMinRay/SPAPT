# SPAPT
Signal Processing for Analog Phaser Tomography.

This is the digital singal processing simulation model for the Analog device cn0566 hybrid phaser kit.

## Project Overview
Our project aims to pioneer the utilization of cost-effective one-dimensional analog phase array technology in radar tomography. By integrating the cn0566 phaser array onto a rotational platform, aligned axis with center of phaser, we aim to pioneer three-dimensional tomography synthesis through inventive signal processing algorithms.
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
SPAPT only needs [numpy](https://pypi.org/project/numpy/), [matplotlib](https://pypi.org/project/matplotlib/) and [scipy](https://pypi.org/project/scipy/).

Simply clone SPAPT form repository.

```bash
git clone https://github.com/TMinRay/SPAPT.git
```

## Usage

To start the simulator
```bash
$ python simradar
```

interactive commands use ` `(space) to differentiate blocks.

## Example Results
Our simulator, emulating real-time system features, employs an infinite loop state machine to generate collected pulse signals for each antenna element. With all 32 element receiving time series simulated, the simulator offering all digital beamforming, simultaneously synthesize multiple steering angles within single dwell, subarray beamforming, in-phase summing timeseries to 8 channels, like cn0566 and steering receiving beams with eight subarray series. The simulator could also emulate the analog imaging with limiting the flexibility of summing phases while beamforming which is exactly operating on cn0566. Preliminary tests yield promising outcomes, showcasing the rotational one-dimensional phaser's potential for three-dimensional tomography synthesis. Optimal dwell time and platform rotation speed prove crucial, with figures demonstrating the imaging of targets featuring varying radar cross-sections (RCS) and displacements. This includes comparisons with all-digital beamforming (considered the golden standard), fixed subarray beamforming, and rotational subarray beamforming.

```SPAPT
# Summary current simulator
$ s
platform
         current orientation     0.0 degree
         platform speed          0.0 dps

transceiver
         n pulse / fire          512
         pulse interval          1
         PRF                     300.0 kHz
         F0                      10.3 GHz

objects
        Range                    [50, 50, 50]
        Vr                       [60, -40, 0]
        RCS                      [2, 5, 10]
        Azimuth                  [0, 90, 45]
        Zenith                   [0, 30, 45]
# fix panel all digital beamforming
$ d a
```
![All_Digital](/image/All_Digital.png)
```SPAPT
# fix panel subarray beamforming
$ d s
```
![Subarray_Fix](/image/Subarray_Fix.png)

```SPAPT
# change to mission 1 (spin panel)
m 1
# spin panel subarray beamforming
$ d s
```
![Subarry_spin](/image/Subarry_spin.png)

```SPAPT
# return to default mission 0
m 0
# plot rdm
d r
```
![rdm0](/image/rdm_0_deg.png)
```SPAPT
# set orienation to 45
p o 45
d r
```
![rdm45](/image/rdm_45_deg.png)
## SPAPT command blocks

| First | second | Parameters | Description |
| --- | --- | --- | --- |
| s | --- | --- | Summary of current radar parameters|
| s |  p  | --- | Summary only platform parameters|
| s |  t  | --- | Summary only transceiver parameters|
| s |  o  | --- | Summary only objects(target) parameters|
| --- | --- | --- | --- |
| d |  a  | --- | Display All digital beamforming result|
| d |  s  | --- | Display Subarray beamforming result|
| d |  r  | --- | Display Range Doppler Map|
| --- | --- | --- | --- |
| m |  Number(0-2)  | --- | Change to predefined scenario (Mission)|
| --- | --- | --- | --- |
| p |  o  | Number | Change Phaser current orientation (deg)|
| p |  s  | Number | Change Phaser roation speed (deg/s)|
| --- | --- | --- | --- |
| t |  n  | Number | Change Numbers of pulses / processing |
| t |  prf  | Number | Change Pulse Repetition Frequency |
| --- | --- | --- | --- |
| q | --- | --- | Turn off simulator |

## Conclusion
Our project has the potential to demonstrate the feasibility of utilizing a one-dimensional phaser array configuration for acquiring three-dimensional tomography. This innovative approach could find applications in space-borne radar systems. The technique presents a significant advantage in terms of weight savings—fewer elements translate to lighter payloads, resulting in substantial budget savings. Remarkably, this is achieved without compromising data quality, making our solution a compelling prospect for cost-effective yet high-quality radar tomography applications, especially in space-based scenarios.

