# SPAPT
Signal Processing for Analog Phaser Tomography.

This is the digital singal processing simulation model for the Analog device cn0566 hybrid phaser kit.

![All_Digital](/image/All_Digital.png)
![Subarray_Fix](/image/Subarray_Fix.png)
![Subarry_spin](/image/Subarry_spin.png)

## Installation
SPAPT only needs [numpy](https://pypi.org/project/numpy/), [matplotlib](https://pypi.org/project/matplotlib/) and [scipy](https://pypi.org/project/scipy/).

Simply clone SPART form repository.

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
To check current platform status.
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
# fix panel subarray beamforming
$ d s

# change to mission 1 (spin panel)
m 1
# spin panel subarray beamforming
$ d s
# spin panel all digital
$ d a

# return to default mission 0
m 0
# plot rdm
d r
# set orienation to 45
p o 45
d r
```

### command blocks

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
