# SPAPT simulator

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