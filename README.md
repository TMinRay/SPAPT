# SPAPT
Signal Processing for Analog Phaser Tomography.

This is the siganl model as well the digital singal processing for the Analog device cn0566 hybrid phaser kit.

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
$ s p
         current orientation     0.0 degree
         platform speed          0.0 dps
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


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
