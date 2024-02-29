#!/usr/bin/env python3
#  Must use Python 3
# Copyright (C) 2022 Analog Devices, Inc.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#     - Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     - Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the
#       distribution.
#     - Neither the name of Analog Devices, Inc. nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#     - The use of this software may or may not infringe the patent rights
#       of one or more patent holders.  This license does not release you
#       from the requirement that you obtain separate licenses from these
#       patent holders to use this software.
#     - Use of the software either in source or binary form, must be run
#       on or directly connected to an Analog Devices Inc. component.
#
# THIS SOFTWARE IS PROVIDED BY ANALOG DEVICES "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED.
#
# IN NO EVENT SHALL ANALOG DEVICES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, INTELLECTUAL PROPERTY
# RIGHTS, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# A minimal example script to demonstrate some basic concepts of controlling
# the Pluto SDR.
# The script reads in the measured HB100 source's frequency (previously stored
# with the find_hb100 utility), sets the phaser's pll such that the HB100's frequency
# shows up at 1MHz, captures a buffer of data, take FFT, plot time and frequency domain

# Since the Pluto is attached to the phaser board, we do have
# to do some setup and housekeeping of the ADAR1000s, but other than that it's
# trimmed down about as much as possible.

# Import libraries.
from time import sleep

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from adi import ad9361
from adi.cn0566 import CN0566

from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtWidgets import *
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import sys
# from phaser_functions import load_hb100_cal, spec_est
# from scipy import signal

# First try to connect to a locally connected CN0566. On success, connect,
# on failure, connect to remote CN0566

try:
    print("Attempting to connect to CN0566 via ip:localhost...")
    my_phaser = CN0566(uri="ip:localhost")
    print("Found CN0566. Connecting to PlutoSDR via default IP address...")
    my_sdr = ad9361(uri="ip:192.168.2.1")
    print("PlutoSDR connected.")

except:
    print("CN0566 on ip.localhost not found, connecting via ip:phaser.local...")
    my_phaser = CN0566(uri="ip:phaser.local")
    print("Found CN0566. Connecting to PlutoSDR via shared context...")
    my_sdr = ad9361(uri="ip:phaser.local:50901")
    print("Found SDR on shared phaser.local.")

def steer_angle_to_phase_diff(th,fc,d):
    return 2*np.pi*d*fc*np.sin(th*np.pi/180)/3e8

my_phaser.sdr = my_sdr  # Set my_phaser.sdr

sleep(0.5)

# By default device_mode is "rx"
my_phaser.configure(device_mode="rx")
my_phaser.load_channel_cal()
my_phaser.load_gain_cal("gain_cal_val.pkl")
my_phaser.load_phase_cal("phase_cal_val.pkl")

# try:
#     my_phaser.SignalFreq = load_hb100_cal()
#     print("Found signal freq file, ", my_phaser.SignalFreq)
# except:
#     my_phaser.SignalFreq = 10.409e9
#     print("No signal freq file found, setting to 10.409 GHz")


# Configure CN0566 parameters.
#     ADF4159 and ADAR1000 array attributes are exposed directly, although normally
#     accessed through other methods.


# Set all antenna elements to half scale - a typical HB100 will have plenty
# of signal power.

gain_list = [84] * 8  # (64 is about half scale)
# gain_list = [8, 34, 84, 127, 127, 84, 34, 8]  # Blackman taper
for i in range(0, len(gain_list)):
    # my_phaser.set_chan_gain(i, gain_list[i], apply_cal=False)
    my_phaser.set_chan_gain(i, gain_list[i])

# Aim the beam at boresight (zero degrees). Place HB100 right in front of array.
my_phaser.set_beam_phase_diff(0.0)

#########
# #  Configure SDR parameters. Start with the more involved settings, don't
# # pay too much attention to these. They are covered in much more detail in
# # Software Defined Radio for Engineers.

# my_sdr._ctrl.debug_attrs["adi,frequency-division-duplex-mode-enable"].value = "1"
# my_sdr._ctrl.debug_attrs[
#     "adi,ensm-enable-txnrx-control-enable"
# ].value = "0"  # Disable pin control so spi can move the states
# my_sdr._ctrl.debug_attrs["initialize"].value = "1"
#########
sample_rate = 0.6e6
center_freq = 2.2e9
signal_freq = 100e3
num_slices = 200
fft_size = 1024 * 16
# fft_size = 1024*8
img_array = np.zeros((num_slices, fft_size))

# Create radio.
# This script is for Pluto Rev C, dual channel setup
my_sdr.sample_rate = int(sample_rate)
# my_sdr.rx_rf_bandwidth = int(10e6)  # Analog bandwidth
# Configure Rx
my_sdr.rx_lo = int(center_freq)  # set this to output_freq - (the freq of the HB100)
# my_sdr.filter = "LTE20_MHz.ftr"  # Handy filter for fairly widdeband measurements
my_sdr.rx_enabled_channels = [0, 1]  # enable Rx1 (voltage0) and Rx2 (voltage1)
my_sdr.rx_buffer_size = int(fft_size)
my_sdr.gain_control_mode_chan0 = "manual"  # manual or slow_attack
my_sdr.gain_control_mode_chan1 = "manual"  # manual or slow_attack
my_sdr.rx_hardwaregain_chan0 = int(30)  # must be between -3 and 70
my_sdr.rx_hardwaregain_chan1 = int(30)  # must be between -3 and 70
# Configure Tx
my_sdr.tx_lo = int(center_freq)
my_sdr.tx_enabled_channels = [0, 1]
my_sdr.tx_cyclic_buffer = True  # must set cyclic buffer to true for the tdd burst mode.  Otherwise Tx will turn on and off randomly
my_sdr.tx_hardwaregain_chan0 = -88  # must be between 0 and -88
my_sdr.tx_hardwaregain_chan1 = -0  # must be between 0 and -88


# my_sdr.rx_enabled_channels = [0, 1]  # enable Rx1 (voltage0) and Rx2 (voltage1)
# my_sdr._rxadc.set_kernel_buffers_count(1)  # No stale buffers to flush
# rx = my_sdr._ctrl.find_channel("voltage0")
# rx.attrs["quadrature_tracking_en"].value = "1"  # enable quadrature tracking
# # Make sure the Tx channels are attenuated (or off) and their freq is far away from Rx
# # this is a negative number between 0 and -88
# my_sdr.tx_hardwaregain_chan0 = int(-80)
# my_sdr.tx_hardwaregain_chan1 = int(-80)


# # These parameters are more closely related to analog radio design
# # and are what you would adjust to change the IFs, signal bandwidths, sample rate, etc.
# #
# # Sample rate is set to 30Msps,
# # for a total of 30MHz of bandwidth (quadrature sampling)
# # Filter is 20MHz LTE, so you get a bit less than 20MHz of usable
# # bandwidth.

# my_sdr.sample_rate = int(30e6)  # Sampling rate
# my_sdr.rx_buffer_size = int(1024)  # Number of samples per buffer
# my_sdr.rx_rf_bandwidth = int(10e6)  # Analog bandwidth

# # Manually control gain - in most applications, you want to enable AGC to keep
# # to adapt to changing conditions. Since we're taking quasi-quantitative measurements,
# # we want to set the gain to a fixed value.
# my_sdr.gain_control_mode_chan0 = "manual"  # DISable AGC
# my_sdr.gain_control_mode_chan1 = "manual"
# my_sdr.rx_hardwaregain_chan0 = 0  # dB
# my_sdr.rx_hardwaregain_chan1 = 0  # dB

# my_sdr.rx_lo = int(2.2e9)  # Downconvert by 2GHz  # Receive Freq
# my_sdr.filter = "LTE20_MHz.ftr"  # Handy filter for fairly widdeband measurements


# Configure the ADF4159 Rampling PLL
output_freq = 10e9 + center_freq
BW = 500e6
num_steps = 1000
ramp_time = 1e3  # us
ramp_time_s = ramp_time / 1e6
my_phaser.frequency = int(output_freq / 4)  # Output frequency divided by 4
my_phaser.freq_dev_range = int(
    BW / 4
)  # frequency deviation range in Hz.  This is the total freq deviation of the complete freq ramp
my_phaser.freq_dev_step = int(
    BW / num_steps / 4
)  # frequency deviation step in Hz.  This is fDEV, in Hz.  Can be positive or negative
my_phaser.freq_dev_time = int(
    ramp_time
)  # total time (in us) of the complete frequency ramp
my_phaser.delay_word = 4095  # 12 bit delay word.  4095*PFD = 40.95 us.  For sawtooth ramps, this is also the length of the Ramp_complete signal
my_phaser.delay_clk = "PFD"  # can be 'PFD' or 'PFD*CLK1'
my_phaser.delay_start_en = 0  # delay start
my_phaser.ramp_delay_en = 0  # delay between ramps.
my_phaser.trig_delay_en = 0  # triangle delay
# my_phaser.ramp_mode = "continuous_sawtooth"  # ramp_mode can be:  "disabled", "continuous_sawtooth", "continuous_triangular", "single_sawtooth_burst", "single_ramp_burst"
my_phaser.ramp_mode = "continuous_triangular"  # ramp_mode can be:  "disabled", "continuous_sawtooth", "continuous_triangular", "single_sawtooth_burst", "single_ramp_burst"

my_phaser.sing_ful_tri = (
    0  # full triangle enable/disable -- this is used with the single_ramp_burst mode
)
my_phaser.tx_trig_en = 0  # start a ramp with TXdata
my_phaser.enable = 0  # 0 = PLL enable.  Write this last to update all the registers

# Print config
print(
    """
CONFIG:
Sample rate: {sample_rate}MHz
Num samples: 2^{Nlog2}
Bandwidth: {BW}MHz
Ramp time: {ramp_time}ms
Output frequency: {output_freq}MHz
IF: {signal_freq}kHz
""".format(
        sample_rate=sample_rate / 1e6,
        Nlog2=int(np.log2(fft_size)),
        BW=BW / 1e6,
        ramp_time=ramp_time / 1e3,
        output_freq=output_freq / 1e6,
        signal_freq=signal_freq / 1e3,
    )
)

# Create a sinewave waveform
fs = int(my_sdr.sample_rate)
print("sample_rate:", fs)
N = int(my_sdr.rx_buffer_size)
fc = int(signal_freq / (fs / N)) * (fs / N)
ts = 1 / float(fs)
t = np.arange(0, N * ts, ts)
i = np.cos(2 * np.pi * t * fc) * 2 ** 14
q = np.sin(2 * np.pi * t * fc) * 2 ** 14
iq = 1 * (i + 1j * q)
# iq = np.ones(i.shape)+1j*np.zeros(i.shape)

# fc = int(300e3 / (fs / N)) * (fs / N)
# i = np.cos(2 * np.pi * t * fc) * 2 ** 14
# q = np.sin(2 * np.pi * t * fc) * 2 ** 14
# iq_300k = 1 * (i + 1j * q)

# Send data
my_sdr._ctx.set_timeout(0)
my_sdr.tx([iq * 0.5, iq])  # only send data to the 2nd channel (that's all we need)
# my_sdr.tx([iq * 0, iq])

c = 3e8
default_rf_bw = 500e6
N_frame = fft_size
freq = np.arange(0,fs,fs/int(N_frame))
# freq = np.linspace(-fs / 2, fs / 2, int(N_frame))

slope = BW / ramp_time_s
dist = (freq - signal_freq) * c / (4 * slope)
# dist = freq * c / (4 * slope)

xdata = freq


font = QtGui.QFont()
font.setPointSize(16)

def get_img_trans(axx,axy):
    dx = (axx[-1]-axx[0])/axx.size
    dy = (axy[-1]-axy[0])/axy.size
    return axx[0], axy[0], dx, dy

title_style = {"size": "20pt"}
label_style = {"color": "#FFF", "font-size": "14pt"}


def nuttall_window(N):
    a=[0.3635819,
    0.4891775,
    0.1365995,
    0.0106411]
    x=np.arange(N)/N*np.pi
    return a[0] - a[1]*np.cos(2*x) + a[2]*np.cos(4*x) - a[3]*np.cos(6*x)

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive FFT")
        self.setGeometry(100, 100, 1800, 1200)
        # self.num_rows = 12
        self.fft_size = fft_size
        self.az = np.arange(-20,21,10)
        # self.az = np.zeros((5))
        Fs = sample_rate
        # self.freq = np.arange(-Fs/2,Fs/2,Fs/self.fft_size)
        self.freq = np.arange(0,Fs,Fs/self.fft_size)
        self.tframe = np.arange(50)
        self.img = np.full((self.az.size, self.tframe.size, self.freq.size),-300)
        self.r_cal = np.zeros((sself.freq.size))[np.newaxis,np.newaxis,:]
        self.offset = -300
        # self.plot_dist = False
        self.UiComponents()
        # showing all the widgets
        self.show()

    # method for components
    def UiComponents(self):
        widget = QWidget()

        # global layout
        layout = QGridLayout()

        # Control Panel
        control_label = QLabel("CN0566 FMCW Radar")
        font = control_label.font()
        font.setPointSize(20)
        control_label.setFont(font)
        control_label.setAlignment(Qt.AlignHCenter)  # | Qt.AlignVCenter)
        layout.addWidget(control_label, 0, 0, 1, 5)

        # Check boxes
        self.x_axis_check = QCheckBox("Toggle Range/Frequency x-axis")
        font = self.x_axis_check.font()
        font.setPointSize(15)
        self.x_axis_check.setFont(font)

        self.x_axis_check.stateChanged.connect(self.change_x_axis)
        layout.addWidget(self.x_axis_check, 2, 0)

        self.r_cal_check = QCheckBox("Toggle Range correction factor")
        font = self.r_cal_check.font()
        self.r_cal_check.setFont(font)

        self.r_cal_check.stateChanged.connect(self.range_correction)
        layout.addWidget(self.r_cal_check, 2, 3)

        # Range resolution
        # Changes with the RF BW slider
        default_rf_bw = BW
        # c = 3e8
        self.range_res_label = QLabel(
            "B<sub>RF</sub>: %0.2f MHz - R<sub>res</sub>: %0.2f m"
            % (default_rf_bw / 1e6, c / (2 * default_rf_bw))
        )
        font = self.range_res_label.font()
        font.setPointSize(15)
        self.range_res_label.setFont(font)
        self.range_res_label.setAlignment(Qt.AlignRight)
        self.range_res_label.setMinimumWidth(300)
        layout.addWidget(self.range_res_label, 4, 1)

        # RF bandwidth slider
        self.bw_slider = QSlider(Qt.Horizontal)
        self.bw_slider.setMinimum(100)
        self.bw_slider.setMaximum(500)
        self.bw_slider.setValue(int(default_rf_bw / 1e6))
        self.bw_slider.setTickInterval(50)
        self.bw_slider.setTickPosition(QSlider.TicksBelow)
        self.bw_slider.valueChanged.connect(self.get_range_res)
        layout.addWidget(self.bw_slider, 4, 0)

        self.set_bw = QPushButton("Set RF Bandwidth")
        self.set_bw.pressed.connect(self.set_range_res)
        layout.addWidget(self.set_bw, 5, 0, 1, 2)

        AZLayout = QHBoxLayout()
        self.az_input = []
        for ip in range(5):
            self.az_input.append(QLineEdit())
            self.az_input[ip].setAlignment(Qt.AlignCenter)
            self.az_input[ip].setStyleSheet("background-color: #444; color: #fff; border: 1px solid #666;")
            self.az_input[ip].setText('{:d}'.format(int(self.az[ip])))
            AZLayout.addWidget(self.az_input[ip])
            self.az_input[ip].setFont(font)
            def update_az_wrapper(text, ip=ip):
                self.update_az(text, ip)
            self.az_input[ip].textChanged.connect(update_az_wrapper)

        # FFT plot
        self.fft_plot = pg.plot()
        self.fft_plot.setMinimumWidth(1200)
        self.fft_plot.setMaximumHeight(300)
        self.fft_curve = self.fft_plot.plot(self.freq, pen="y", width=6)

        self.fft_plot.setLabel("bottom", text="Frequency", units="Hz", **label_style)
        self.fft_plot.setLabel("left", text="Magnitude", units="dB", **label_style)
        self.fft_plot.setTitle("Received Signal - Frequency Spectrum", **title_style)
        self.fft_plot.setYRange(-45, 5)
        self.fft_plot.setXRange(self.freq[0],self.freq[-1]/2)
        layout.addWidget(self.fft_plot, 6, 0, 2, 5)

        layout.addLayout(AZLayout, 8, 0, 1, 5)
        self.plot_xaxis = self.freq
        
        # Waterfall plot
        # self.waterfall = pg.PlotWidget()
        self.gr_wid = pg.GraphicsLayoutWidget()
        self.waterfall = []
        self.imageitem = []
        for ip in range(5):
            self.waterfall.append( self.gr_wid.addPlot() )
        # self.waterfall = self.gr_wid.addPlot()
            self.imageitem.append( pg.ImageItem() )
            self.waterfall[ip].addItem(self.imageitem[ip])
            self.set_Quads(self.imageitem[ip])
        
        # self.imageitem1 = pg.ImageItem()
        # self.waterfall1.addItem(self.imageitem1)
            # zoom_freq = 40e3
            # self.waterfall[ip].setRange(xRange=(self.az[0], self.az[-1] ), yRange=(self.freq[0],self.freq[-1]))
            self.waterfall[ip].setRange(xRange=(self.tframe[0], self.tframe[-1] ), yRange=(self.freq[0],self.freq[-1]/2))
            self.waterfall[ip].setTitle("Waterfall AZ {:d}".format(int(self.az[ip])), **title_style)
            self.waterfall[ip].setLabel("left", "Frequency", units="Hz", **label_style)
            # self.waterfall[ip].setLabel("bottom", "Azimuth", units="<html><sup>o</sup></html>", **label_style)
            self.waterfall[ip].setLabel("bottom", "Frame", **label_style)
            self.waterfall[ip].getAxis("bottom").setTickFont(font)
            self.waterfall[ip].getAxis("left").setTickFont(font)
        # self.waterfall.setTickFont
        bar = pg.ColorBarItem(
            values = (-45, 5),
            colorMap='CET-L4',
            label='horizontal color bar',
            limits = (None, None),
            rounding=0.1,
            orientation = 'h',
            pen='#8888FF', hoverPen='#EEEEFF', hoverBrush='#EEEEFF80'
        )
        # bar.setImageItem( self.imageitem, insert_in=self.waterfall )
        bar.setImageItem( self.imageitem )
        self.gr_wid.addItem(bar, 1, 0, 1, 5)
        layout.addWidget(self.gr_wid, 11, 0, 18, 5)

        # self.br_wid = pg.GraphicsLayoutWidget()
        # self.cb = self.br_wid.addPlot()
        # self.cb.addItem(bar)

        # layout.addWidget(self.br_wid, 19, 0, 1, 5)
        # # self.img_array = np.zeros((num_slices, fft_size))

        widget.setLayout(layout)
        # setting this widget as central widget of the main window
        self.setCentralWidget(widget)

    def set_Quads(self, im):
        tr = QtGui.QTransform()
        # trans_para = get_img_trans(self.az,self.plot_xaxis)
        trans_para = get_img_trans(self.tframe,self.plot_xaxis)
        tr.translate(trans_para[0], trans_para[1])
        tr.scale(trans_para[2], trans_para[3])
        im.setTransform(tr)

    @pyqtSlot(str)
    def update_az(self, text, ix):
        # print(text)
        if is_float(text):
            self.az[ix] = float(text)
            self.waterfall[ix].setTitle("Waterfall AZ {:d}".format(int(self.az[ix])), **title_style)
        else:
            print(is_float(text))

    def get_range_res(self):
        """ Updates the slider bar label with RF bandwidth and range resolution
        Returns:
            None
        """
        bw = self.bw_slider.value() * 1e6
        range_res = c / (2 * bw)
        self.range_res_label.setText(
            "B<sub>RF</sub>: %0.2f MHz - R<sub>res</sub>: %0.2f m"
            % (bw / 1e6, c / (2 * bw))
        )

    # def get_water_levels(self):
    #     """ Updates the waterfall intensity levels
    #     Returns:
    #         None
    #     """
    #     if self.low_slider.value() > self.high_slider.value():
    #         self.low_slider.setValue(self.high_slider.value())
    #     self.low_label.setText("LOW LEVEL: %0.0f" % (self.low_slider.value()))
    #     self.high_label.setText("HIGH LEVEL: %0.0f" % (self.high_slider.value()))

    # def get_steer_angle(self):
    #     """ Updates the steering angle readout
    #     Returns:
    #         None
    #     """
    #     self.steer_label.setText("%0.0f DEG" % (self.steer_slider.value()))
    #     phase_delta = (
    #         2
    #         * 3.14159
    #         * 10.25e9
    #         * 0.014
    #         * np.sin(np.radians(self.steer_slider.value()))
    #         / (3e8)
    #     )
    #     my_phaser.set_beam_phase_diff(np.degrees(phase_delta))

    def set_range_res(self):
        """ Sets the RF bandwidth
        Returns:
            None
        """
        # global slope
        bw = self.bw_slider.value() * 1e6
        slope = bw / ramp_time_s
        dist = (freq - signal_freq) * c / (4 * slope)
        print("New slope: %0.2fMHz/s" % (slope / 1e6))
        # if self.x_axis_check.isChecked() == True:
        #     print("Range axis")
        #     # self.plot_dist = True
        #     range_x = (100e3) * c / (4 * slope)
        #     self.fft_plot.setXRange(0, range_x)
        # else:
        #     print("Frequency axis")
        #     # self.plot_dist = False
        #     self.fft_plot.setXRange(self.freq[0],self.freq[-1]/2)
        my_phaser.freq_dev_range = int(bw / 4)  # frequency deviation range in Hz
        my_phaser.freq_dev_step = int(
            BW / num_steps/ 4
        )
        my_phaser.enable = 0
        self.change_x_axis(self.x_axis_check.isChecked())
        self.img = np.full((win.az.size, win.tframe.size, win.freq.size),-300)

    def get_dist(self):
        bw = self.bw_slider.value() * 1e6
        slope = bw / ramp_time_s
        dist = (self.freq - signal_freq) * c / (4 * slope)
        return dist

    def change_x_axis(self, state):
        """ Toggles between showing frequency and range for the x-axis
        Args:
            state (QtCore.Qt.Checked) : State of check box
        Returns:
            None
        """
        # global slope
        # plot_state = win.fft_plot.getViewBox().state
        # if state == QtCore.Qt.Checked:
        #     print("Range axis")
        #     plot_dist = True
        #     range_x = (100e3) * c / (4 * slope)
        #     self.fft_plot.setXRange(0, range_x)
        # else:
        #     print("Frequency axis")
        #     plot_dist = False
        #     self.fft_plot.setXRange(100e3, 200e3)
        dist = self.get_dist()
        # print("New slope: %0.2fMHz/s" % (slope / 1e6))
        if self.x_axis_check.isChecked() == True:
            print("Range axis")
            # self.plot_dist = True
            range_x = np.max(self.freq - signal_freq) * c / (4 * slope)
            self.plot_xaxis = dist
            self.fft_plot.setTitle("Received Signal - Range", **title_style)
            self.fft_plot.setLabel("bottom", text="Range", units="m")
            # self.fft_plot.setXRange(0, range_x/2)
            self.fft_plot.setXRange(0, 5)
            for ip in range(5):
                # self.waterfall[ip].setRange(yRange=(0, range_x/2))
                self.waterfall[ip].setRange(yRange=(0, 5))
                self.waterfall[ip].setLabel("left", "Range", units="m")
                self.set_Quads(self.imageitem[ip])
        else:
            print("Frequency axis")
            # self.plot_dist = False
            self.plot_xaxis = self.freq
            # self.fft_plot.setXRange(0, 200e3)
            self.fft_plot.setXRange(self.freq[0],self.freq[-1]/2)
            self.fft_plot.setTitle("Received Signal - Frequency Spectrum", **title_style)
            self.fft_plot.setLabel("bottom", text="Frequency", units="Hz")
            for ip in range(5):
                self.waterfall[ip].setRange(yRange=(self.freq[0],self.freq[-1]/2))
                self.waterfall[ip].setLabel("left", "Frequency", units="Hz")
                self.set_Quads(self.imageitem[ip])

    def range_correction(self, state):
        if state == QtCore.Qt.Checked:
            self.r_cal = np.zeros((sself.freq.size))[np.newaxis,np.newaxis,:]
        else:
            self.r_cal = np.full((sself.freq.size),-999)
            dist = self.get_dist()
            self.r_cal[dist>0] = 40*np.log10(dist[dist>0])
            self.r_cal = self.r_cal[np.newaxis,np.newaxis,:]


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        print(value)
        return False


# create pyqt5 app
App = QApplication(sys.argv)

# create the instance of our Window
win = Window()
index = 0

ref = 2 ** 12
# win.img = np.full((win.az.size, win.tframe.size, win.freq.size),-300)

def update():
    frdata = np.zeros((win.freq.size),dtype=np.complex_)
    fxdata = np.zeros((win.az.size, win.freq.size),dtype=np.complex_)
    # win_funct = np.blackman(win.freq.size)
    win_funct = nuttall_window(win.freq.size)
    for avgpulse in range(1):
        for ia, steer in enumerate(win.az):
            my_phaser.set_beam_phase_diff(steer_angle_to_phase_diff(steer, output_freq,0.014)*180/np.pi)
            # sleep(5e-2)
            for i in range(4):
                data = my_sdr.rx()
            # N = win.fft_size
            # t = np.arange(N)/sample_rate
            # data = [np.exp(2j*np.pi*(50e3*t**2+(30+steer)*1e4*t)),np.exp(2j*np.pi*(50e3*t**2+(30+steer)*1e4*t))]
            data_sum = data[0] + data[1]
            fxdata[ia,:] = 1 / N * np.fft.fft(data_sum * win_funct)
            frdata += 1 / N * np.fft.fft(data_sum * win_funct)
    ampl = np.abs(fxdata)
    ampl = 20 * np.log10(ampl / ref + 10 ** -20)
    win.img = np.roll( win.img, 1, axis=1 )
    win.img[:,0,:] = ampl
    pr = np.abs(frdata)
    pr = 20 * np.log10(pr / ref + 10 ** -20)
    pr = pr - np.max(pr)
    if np.abs(win.offset-np.max(win.img))>5:
        win.offset=np.max(win.img)
    for ip in range(5):
        win.imageitem[ip].setImage(win.img[ip,:,:] - win.offset + win.r_cal, autoLevels=False)
    win.fft_curve.setData(win.plot_xaxis, pr)
    # win.fft_plot.setLabel("bottom", text="Frequency", units="Hz", **label_style)


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

# start the app
sys.exit(App.exec())



# Clean up / close connections
del my_sdr
del my_phaser
