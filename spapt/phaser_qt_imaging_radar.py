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
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import adi
from adi import ad9361
from adi.cn0566 import CN0566

from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtWidgets import *
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import sys
import socket

# First try to connect to a locally connected CN0566. On success, connect,
# on failure, connect to remote CN0566

# try:
#     print("Attempting to connect to CN0566 via ip:localhost...")
#     my_phaser = CN0566(uri="ip:localhost")
#     print("Found CN0566. Connecting to PlutoSDR via default IP address...")
#     # my_sdr = ad9361(uri="ip:192.168.2.1")
#     sdr_ip = "ip:192.168.2.1"

# except:
#     print("CN0566 on ip.localhost not found, connecting via ip:phaser.local...")
#     my_phaser = CN0566(uri="ip:phaser.local")
#     print("Found CN0566. Connecting to PlutoSDR via shared context...")
#     sdr_ip = "ip:phaser.local:50901"
#     print("SDR on shared phaser.local.")

print("Connecting CN0566 via ip:phaser.local...")
my_phaser = CN0566(uri="ip:phaser.local")
print("Found CN0566. Connecting to PlutoSDR via shared context...")
sdr_ip = "ip:phaser.local:50901"
print("SDR on shared phaser.local.")
my_sdr = ad9361(uri=sdr_ip)
print("PlutoSDR connected.")

my_phaser.sdr = my_sdr  # Set my_phaser.sdr

time.sleep(0.5)

# By default device_mode is "rx"
my_phaser.configure(device_mode="rx")
my_phaser.load_channel_cal()
my_phaser.load_gain_cal("gain_cal_val.pkl")
my_phaser.load_phase_cal("phase_cal_val.pkl")

# Configure CN0566 parameters.
#     ADF4159 and ADAR1000 array attributes are exposed directly, although normally
#     accessed through other methods.

# Set all antenna elements to 84/ 128.
gain_list = [84] * 8  # (64 is about half scale)
# gain_list = [8, 34, 84, 127, 127, 84, 34, 8]  # Blackman taper
for i in range(0, len(gain_list)):
    my_phaser.set_chan_gain(i, gain_list[i])

# Reset pointing of the beam at boresight (zero degrees).
my_phaser.set_beam_phase_diff(0.0)

# Setup Raspberry Pi GPIO states
try:
    my_phaser._gpios.gpio_tx_sw = 0  # 0 = TX_OUT_2, 1 = TX_OUT_1
    my_phaser._gpios.gpio_vctrl_1 = 1 # 1=Use onboard PLL/LO source  (0=disable PLL and VCO, and set switch to use external LO input)
    my_phaser._gpios.gpio_vctrl_2 = 1 # 1=Send LO to transmit circuitry  (0=disable Tx path, and send LO to LO_OUT)
except:
    my_phaser.gpios.gpio_tx_sw = 0  # 0 = TX_OUT_2, 1 = TX_OUT_1
    my_phaser.gpios.gpio_vctrl_1 = 1 # 1=Use onboard PLL/LO source  (0=disable PLL and VCO, and set switch to use external LO input)
    my_phaser.gpios.gpio_vctrl_2 = 1 # 1=Send LO to transmit circuitry  (0=disable Tx path, and send LO to LO_OUT)

#########
# # this part cause restart of phaser trouble (need reboot)
# #  Configure SDR parameters. Start with the more involved settings, don't
# # pay too much attention to these. They are covered in much more detail in
# # Software Defined Radio for Engineers.

# my_sdr._ctrl.debug_attrs["adi,frequency-division-duplex-mode-enable"].value = "1"
# my_sdr._ctrl.debug_attrs[
#     "adi,ensm-enable-txnrx-control-enable"
# ].value = "0"  # Disable pin control so spi can move the states
# my_sdr._ctrl.debug_attrs["initialize"].value = "1"
#########
c = 3e8
sample_rate = 0.6e6
center_freq = 2.2e9
signal_freq = 100e3
CPI_pulse = 1

# Create radio.
# This script is for Pluto Rev C, dual channel setup
my_sdr.sample_rate = int(sample_rate)
# my_sdr.rx_rf_bandwidth = int(10e6)  # Analog bandwidth
# Configure Rx
my_sdr.rx_lo = int(center_freq)  # set this to output_freq - (the freq of the HB100)
# my_sdr.filter = "LTE20_MHz.ftr"  # Handy filter for fairly widdeband measurements
my_sdr.rx_enabled_channels = [0, 1]  # enable Rx1 (voltage0) and Rx2 (voltage1)
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
my_phaser.ramp_mode = "single_sawtooth_burst"  # ramp_mode can be:  "disabled", "continuous_sawtooth", "continuous_triangular", "single_sawtooth_burst", "single_ramp_burst"
my_phaser.sing_ful_tri = (
    0  # full triangle enable/disable -- this is used with the single_ramp_burst mode
)
my_phaser.tx_trig_en = 1  # start a ramp with TXdata
my_phaser.enable = 0  # 0 = PLL enable.  Write this last to update all the registers

# %%
# Configure TDD controller
sdr_pins = adi.one_bit_adc_dac(sdr_ip)
sdr_pins.gpio_tdd_ext_sync = True # If set to True, this enables external capture triggering using the L24N GPIO on the Pluto.  When set to false, an internal trigger pulse will be generated every second
tdd = adi.tddn(sdr_ip)
sdr_pins.gpio_phaser_enable = True
tdd.enable = False         # disable TDD to configure the registers
tdd.sync_external = True
tdd.startup_delay_ms = 1
tdd.frame_length_ms = ramp_time/1e3 + 0.2    # each GPIO toggle is spaced this far apart
tdd.burst_count = CPI_pulse       # number of chirps in one continuous receive buffer

tdd.out_channel0_enable = True
tdd.out_channel0_polarity = False
tdd.out_channel0_on_ms = 0.01    # each GPIO pulse will be 100us (0.6ms - 0.5ms).  And the first trigger will happen 0.5ms into the buffer
tdd.out_channel0_off_ms = 0.2
tdd.out_channel1_enable = True
tdd.out_channel1_polarity = False
tdd.out_channel1_on_ms = 0
tdd.out_channel1_off_ms = 0.1
tdd.out_channel2_enable = False
tdd.enable = True

frame_time = tdd.frame_length_ms*tdd.burst_count   # time in ms
print("frame_time:  ", frame_time, "ms")
buffer_time = 0
power=8
while frame_time > buffer_time:     
    power=power+1
    buffer_size = int(2**power) 
    buffer_time = buffer_size/my_sdr.sample_rate*1000   # buffer time in ms
    if power==23:
        break     # max pluto buffer size is 2**23, but for tdd burst mode, set to 2**22
print("buffer_size:", buffer_size)
my_sdr.rx_buffer_size = buffer_size
print("buffer_time:", buffer_time, " ms")  
PRI = tdd.frame_length_ms / 1e3
PRF = 1 / PRI
fft_size = buffer_size
# First ramp starts with some offset (as defined in the TDD section above)
start_offset_time = tdd.out_channel0_on_ms/1e3

# From start of each ramp, how many "good" points do we want?
# For best freq linearity, stay away from the start of the ramps
begin_offset_time = 0.02e-3
good_ramp_time = ramp_time_s - begin_offset_time
good_ramp_samples = int(good_ramp_time * sample_rate)
start_offset_samples = int((start_offset_time+begin_offset_time)*sample_rate)

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
iq = 0.9 * (i + 1j * q)

my_sdr._ctx.set_timeout(30000)
my_sdr._rx_init_channels() 
# Send data
my_sdr.tx([iq, iq])      # send data to the both channel (it should send only one
                         # with my_sdr.tx([iq, iq *0 ])
                         # However, there maybe a glitch while enable TDD engine,
                         # it sometime doesn't transmit if only send one channel.)
                         # same time the base band signal is not correctly send out 
                         # double check the recevied signal is at correct frequency
                         # temporary send TX in both channel to go around bug.

def steer_angle_to_phase_diff(th,fc,d):
    return 2*np.pi*d*fc*np.sin(th*np.pi/180)/3e8

def get_img_trans(axx,axy):
    dx = (axx[-1]-axx[0])/axx.size
    dy = (axy[-1]-axy[0])/axy.size
    return axx[0], axy[0], dx, dy

def rotation_x(ang):
    ang = np.deg2rad(ang)
    return np.array([[1,0,0],[0,np.cos(ang),-np.sin(ang)],[0,np.sin(ang),np.cos(ang)]])

def rotation_z(ang):
    ang = np.deg2rad(ang)
    return np.array([[np.cos(ang),-np.sin(ang),0],[np.sin(ang),np.cos(ang),0],[0,0,1]])

def com(HOST, PORT, TIMEOUT, msg):
    try:
        # Create a socket object
        s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Set the socket timeout
        s.settimeout(TIMEOUT)
        s.connect((HOST, PORT))
        s.sendall(msg)
        # Receive data from the server
        data = s.recv(1024)
        # print("Received:", data.decode())
        s.close()
    except socket.timeout:
        print("Timeout: No data received within {} seconds.".format(TIMEOUT))

    except ConnectionRefusedError:
        print("Connection refused: Unable to connect to the server.")

    except Exception as e:
        print("An error occurred:", e)
    return data

def nuttall_window(N):
    a=[0.3635819,
    0.4891775,
    0.1365995,
    0.0106411]
    x=np.arange(N)/N*np.pi
    return a[0] - a[1]*np.cos(2*x) + a[2]*np.cos(4*x) - a[3]*np.cos(6*x)

title_style = {"size": "20pt"}
label_style = {"color": "#FFF", "font-size": "12pt"}

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CN0566 FMCW Imaging Radar")
        self.HOST = 'localhost'  # IP address of the windows server
        self.PORT = 7727        # Port number of the server
        self.TIMEOUT = 2        # Timeout value in seconds
        self.setGeometry(20, 20, 1706, 960)
        # self.setGeometry(20, 20, 972, 1440)
        self.fft_size = fft_size

        self.az = np.arange(-30,31,2.5)           # exact analog beamforming angle
        self.fan_az = np.arange(-30,30.5,2.5)  # desired beamforming angle (hybrid)
        self.wf_az = np.arange(-15,16,15)       # desired waterfall display angle
        self.wfid = np.zeros(self.wf_az.size,dtype=np.int_) # desired waterfall angle index

        self.freq = np.arange(0,sample_rate,sample_rate/self.fft_size) # frequency axis
                                        # won't fftshift in following processing
                                        # should be fine while signal frequency is positive.
        self.tframe = np.arange(50)             # waterfall record frames
        self.clear_img()
        self.fan = np.full((self.fan_az.size, self.freq.size),-300)
        self.r_cal = np.zeros((self.freq.size))[np.newaxis,:]
        self.offset = -50
        tex,tey = np.meshgrid(np.arange(-30,31,1),np.arange(-29,30,1))
        self.thetax = tex               # desired SAR angle in x
        self.thetay = tey               # desired SAR angle in y
        thetaaz = np.arctan2(np.sin(np.deg2rad(tex)),np.sin(np.deg2rad(tey)))
        thetaele = np.arccos(np.sqrt(1-np.sin(np.deg2rad(tex))**2-np.sin(np.deg2rad(tey))**2))
        self.gar = np.array([np.sin(thetaele)*np.sin(thetaaz), np.sin(thetaele)*np.cos(thetaaz), np.cos(thetaele)])
        self.fan_ar = np.array([np.zeros(self.fan_az.size), np.sin(np.deg2rad(self.fan_az)), np.cos(np.deg2rad(self.fan_az))])
        self.de = np.transpose( np.array([[0,1,0],[0,-1,0]]) )  # effective element location in unit of wavelength
        self.vol_r = [2,3] # 2 3 m
        self.vol_ind = [198,208] # 2 3 m just a default will be overwrite while UiComponents initialize
        self.use_real_ori = False
        self.reset_fa()

        self.cur_time = time.time()
        self.platform_dps = 20
        self.UiComponents()
        self.show()

    # coordinate system
    #       X---ori(xy)
    #       ^  /
    #       | /
    #       |/
    #       z(in)------>y   steer(zy)
    #================================
    #       ^
    #       |
    #       thetax
    #       |
    #       |
    #       o----thetay--->

    def update_ori(self):
        newt = time.time()
        dt = newt - self.cur_time
        if self.use_real_ori:
            self.cur_ori = self.get_ori_from_solo()
        else:
            self.cur_ori = (self.cur_ori - dt*self.platform_dps) % 360
        self.cur_time = newt

    def change_ori_mode(self):
        if self.use_real_ori:
            self.use_real_ori = False
            self.ori_bt.setText("Use real orientation.")
        else:
            self.use_real_ori = True
            self.ori_bt.setText("Use software orientation.")

    def get_ori_from_solo(self):
        posstr = com(self.HOST, self.PORT, self.TIMEOUT, b'cn0566 Request solo position.')
        return ((22.5-float(posstr.decode().split(',')[1]))%360)

    def DBF(self,steer):
        a0 = np.matmul( rotation_z(self.cur_ori) , np.matmul( rotation_x(-steer) , np.array([[0],[0],[1]])))[:,:,np.newaxis]
        ar = self.gar - a0
        element_phase = np.sum(ar[:,np.newaxis,:,:]*np.matmul( rotation_z(self.cur_ori) , self.de)[:,:,np.newaxis,np.newaxis],axis=0)
        return element_phase

    def DBF1d(self,steer):
        a0 = np.matmul( rotation_x(-steer) , np.array([[0],[0],[1]]))
        ar = self.fan_ar - a0
        element_phase = np.sum(ar[:,np.newaxis,:]*self.de[:,:,np.newaxis],axis=0)
        return element_phase

    # method for components
    def UiComponents(self):
        widget = QWidget()

        # global layout
        layout = QGridLayout()

        suptitle_label = QLabel("CN0566 FMCW Imaging Radar")
        font = suptitle_label.font()
        font.setPointSize(24)
        suptitle_label.setFont(font)
        suptitle_label.setAlignment(Qt.AlignHCenter)  # | Qt.AlignVCenter)

        # Check boxes
        self.x_axis_check = QCheckBox("Toggle Range/Frequency x-axis")
        font = self.x_axis_check.font()
        font.setPointSize(17)
        self.x_axis_check.setFont(font)

        self.x_axis_check.stateChanged.connect(self.change_x_axis)

        self.r_cal_check = QCheckBox("Toggle Range correction factor")
        font = self.r_cal_check.font()
        font.setPointSize(17)
        self.r_cal_check.setFont(font)

        self.r_cal_check.stateChanged.connect(self.range_correction)

        # Range resolution
        # Changes with the RF BW slider
        default_rf_bw = BW
        self.range_res_label = QLabel(
            "B<sub>RF</sub>: %0.2f MHz - R<sub>res</sub>: %0.2f m"
            % (default_rf_bw / 1e6, c / (2 * default_rf_bw))
        )
        font = self.range_res_label.font()
        font.setPointSize(15)
        self.range_res_label.setFont(font)
        self.range_res_label.setAlignment(Qt.AlignRight)
        self.range_res_label.setMinimumWidth(150)

        # RF bandwidth slider
        self.bw_slider = QSlider(Qt.Horizontal)
        self.bw_slider.setMinimum(100)
        self.bw_slider.setMaximum(500)
        self.bw_slider.setValue(int(default_rf_bw / 1e6))
        self.bw_slider.setTickInterval(50)
        self.bw_slider.setTickPosition(QSlider.TicksBelow)
        self.bw_slider.valueChanged.connect(self.get_range_res)

        self.set_bw = QPushButton("Set RF Bandwidth")
        self.set_bw.pressed.connect(self.set_range_res)

        self.set_dp = QPushButton("Reset Display Range")
        self.set_dp.pressed.connect(self.reset_display_range)

        self.clear_fa = QPushButton("Reset 2-D imaging integration.")
        self.clear_fa.pressed.connect(self.reset_fa)

        self.integral_num = QLabel("{:d} scans integrated.".format(self.vol_int))

        self.ori_bt = QPushButton("Use real orientation.")
        self.ori_bt.pressed.connect(self.change_ori_mode)

        self.ori_dis = QLabel("current orientation {:.3f} <html><sup>o</sup></html>".format(self.cur_ori))

        for qtres in [self.set_bw, self.set_dp, self.clear_fa, self.integral_num, self.ori_bt, self.ori_dis]:
            font = qtres.font()
            font.setPointSize(15)
            qtres.setFont(font)

        AZLayout = QHBoxLayout()
        self.az_input = []
        for ip in range(3):
            self.az_input.append(QLineEdit())
            self.az_input[ip].setAlignment(Qt.AlignCenter)
            self.az_input[ip].setStyleSheet("background-color: #444; color: #fff; border: 1px solid #666;")
            AZLayout.addWidget(self.az_input[ip])
            self.az_input[ip].setFont(font)
            def update_az_wrapper(text, ip=ip):
                self.update_az(text, ip)
            self.az_input[ip].textChanged.connect(update_az_wrapper)

        # FFT plot
        self.fft_plot = pg.plot()
        self.fft_plot.setMinimumWidth(200)
        self.fft_plot.setMinimumHeight(100)
        self.fft_curve = self.fft_plot.plot(self.freq, pen="y", width=6)

        self.fft_plot.setLabel("bottom", text="Frequency", units="Hz", **label_style)
        self.fft_plot.setLabel("left", text="Mag.", units="dB", **label_style)
        self.fft_plot.setTitle("Received Signal - Frequency Spectrum", **title_style)
        self.fft_plot.setYRange(-55, 5)
        self.fft_plot.setXRange(self.freq[0],self.freq[-1]/2)

        self.plot_xaxis = self.freq
        
        self.fan_wid = pg.GraphicsLayoutWidget()
        self.fanaxs = self.fan_wid.addPlot()
        self.fan_wid.setMaximumWidth(600)
        self.fan_wid.setMaximumHeight(800)
        self.fanimage = pg.ImageItem()
        self.set_Quads(self.fanimage, plot_az=True)
        self.fanaxs.setRange(xRange=(self.fan_az[0], self.fan_az[-1] ), yRange=(self.freq[0],self.freq[-1]/2))
        self.fanaxs.setTitle("Realtime Imaging", **title_style)
        self.fanaxs.setLabel("left", "Frequency", units="Hz", **label_style)
        self.fanaxs.setLabel("bottom", "Azimuth", units="<html><sup>o</sup></html>", **label_style)
        self.fanaxs.getAxis("bottom").setTickFont(font)
        self.fanaxs.getAxis("left").setTickFont(font)
        self.fanaxs.addItem(self.fanimage)

        RLayout = QHBoxLayout()
        self.r_input = []
        for ip in range(2):
            self.r_input.append(QLineEdit())
            self.r_input[ip].setAlignment(Qt.AlignCenter)
            self.r_input[ip].setStyleSheet("background-color: #444; color: #fff; border: 1px solid #666;")
            RLayout.addWidget(self.r_input[ip])
            self.r_input[ip].setFont(font)
            def update_r_wrapper(text, ip=ip):
                self.update_r(text, ip)
            self.r_input[ip].textChanged.connect(update_r_wrapper)

        self.vol_wid = pg.GraphicsLayoutWidget()
        self.vol_wid.setMinimumHeight(450)
        self.volplot = []
        self.volitem = []
        dist=self.get_dist()
        for ip in range(2):
            self.volplot.append( self.vol_wid.addPlot() )
        # self.waterfall = self.gr_wid.addPlot()
            self.volitem.append( pg.ImageItem() )
            self.volplot[ip].addItem(self.volitem[ip])
            self.set_vol_Quads(self.volitem[ip])
            self.volplot[ip].setRange(xRange=(self.thetay[0,0], self.thetay[-1,0] ), yRange=(self.thetax[0,0],self.thetax[0,-1]))
            self.volplot[ip].setTitle("SAR imaging @ {:.1f} m".format(float(dist[self.vol_ind[ip]])), **title_style)
            self.volplot[ip].setLabel("left", "thetax", units="<html><sup>o</sup></html>", **label_style)
            self.volplot[ip].setLabel("bottom", "thetay", units="<html><sup>o</sup></html>", **label_style)
            self.volplot[ip].getAxis("bottom").setTickFont(font)
            self.volplot[ip].getAxis("left").setTickFont(font)

        # Waterfall plot

        self.gr_wid = pg.GraphicsLayoutWidget()
        self.waterfall = []
        self.imageitem = []
        for ip in range(3):
            self.waterfall.append( self.gr_wid.addPlot() )
            self.imageitem.append( pg.ImageItem() )
            self.waterfall[ip].addItem(self.imageitem[ip])
            self.set_Quads(self.imageitem[ip])
            self.waterfall[ip].setRange(xRange=(self.tframe[0], self.tframe[-1] ), yRange=(self.freq[0],self.freq[-1]/2))
            self.waterfall[ip].setTitle("Waterfall AZ {:d}".format(int(self.wf_az[ip])), **title_style)
            self.waterfall[ip].setLabel("left", "Frequency", units="Hz", **label_style)
            self.waterfall[ip].setLabel("bottom", "Frame", **label_style)
            self.waterfall[ip].getAxis("bottom").setTickFont(font)
            self.waterfall[ip].getAxis("left").setTickFont(font)
        self.imageitem.append( self.fanimage )
        for ip in range(2):
            self.imageitem.append( self.volitem[ip] )
        # self.waterfall.setTickFont
        cm = pg.colormap.get('nipy_spectral', source='matplotlib', skipCache=True)
        bar = pg.ColorBarItem(
            values = (-35, -10),
            colorMap=cm,
            limits = (None, None),
            rounding=0.1,
            orientation = 'h',
            pen='#8888FF', hoverPen='#EEEEFF', hoverBrush='#EEEEFF80'
        )
        bar.setImageItem( self.imageitem )
        self.bar_wid = pg.GraphicsLayoutWidget()
        self.bar_wid.addItem(bar, 0, 0, 1, 5)
        self.bar_wid.setMaximumHeight(80)

        for ip in range(3):
            self.az_input[ip].setText('{:d}'.format(int(self.wf_az[ip])))
        for ip in range(2):
            self.r_input[ip].setText('{:d}'.format(int(self.vol_r[ip])))

        layout.addWidget(suptitle_label, 0, 0, 1, 5)
        layout.addWidget(self.fan_wid, 1, 0, 4, 1)
        btlayout = QGridLayout()
        btlayout.addWidget(self.fft_plot, 0, 0, 3, 2)
        btlayout.addWidget(self.x_axis_check, 3, 0)
        btlayout.addWidget(self.r_cal_check, 3, 1)
        btlayout.addWidget(self.range_res_label, 4, 1)
        btlayout.addWidget(self.bw_slider, 4, 0)
        btlayout.addWidget(self.set_dp, 5, 0, 1, 1)
        btlayout.addWidget(self.set_bw, 5, 1, 1, 1)
        btlayout.addWidget(self.ori_bt, 6, 0)
        btlayout.addWidget(self.ori_dis, 6, 1)
        btlayout.addWidget(self.clear_fa, 7, 0)
        btlayout.addWidget(self.integral_num, 7, 1)
        layout.addLayout(btlayout, 1, 1, 1, 1)

        tab = QTabWidget()
        waterfall_page = QWidget(self)
        wf_layout = QGridLayout()
        waterfall_page.setLayout(wf_layout)
        wf_layout.addLayout(AZLayout,0,0,1,5)
        wf_layout.addWidget(self.gr_wid, 1,0,18,5)
        vol_page = QWidget(self)
        vol_layout = QGridLayout()
        vol_page.setLayout(vol_layout)
        vol_layout.addLayout(RLayout,0,0,1,5)
        vol_layout.addWidget(self.vol_wid, 1,0,18,5)
        font = tab.tabBar().font()
        font.setPointSize(20)
        tab.tabBar().setFont(font)
        tab.tabBar().setExpanding(True)
        tab.addTab(waterfall_page, 'waterfall')
        tab.addTab(vol_page, 'SAR imaging')
        tab.setTabShape(QTabWidget.Triangular)
        layout.addWidget(tab,2,1,3,1)
        layout.addWidget(self.bar_wid, 6, 0, 1, 2)

        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def set_vol_Quads(self, im):
        tr = QtGui.QTransform()
        trans_para = get_img_trans(self.thetay[:,0],self.thetax[0,:])
        tr.translate(trans_para[0], trans_para[1])
        tr.scale(trans_para[2], trans_para[3])
        im.setTransform(tr)

    def set_Quads(self, im, plot_az = False):
        tr = QtGui.QTransform()
        if plot_az:
            trans_para = get_img_trans(self.fan_az,self.plot_xaxis)
        else:
            trans_para = get_img_trans(self.tframe,self.plot_xaxis)
        tr.translate(trans_para[0], trans_para[1])
        tr.scale(trans_para[2], trans_para[3])
        im.setTransform(tr)

    @pyqtSlot(str)
    def update_az(self, text, ix):
        # print(text)
        if is_float(text):
            self.wfid[ix] = self.get_wfid(float(text))
            self.wf_az[ix] = self.fan_az[self.wfid[ix]]
            self.waterfall[ix].setTitle("Waterfall AZ {:d}".format(int(self.wf_az[ix])), **title_style)
        else:
            print(is_float(text))

    @pyqtSlot(str)
    def update_r(self, text, ix):
        # print(text)
        dist = self.get_dist()
        if is_float(text):
            self.vol_ind[ix] = self.get_vol_rid(float(text))
            self.vol_r[ix] = dist[self.vol_ind[ix]]
            self.volplot[ix].setTitle("SAR imaging @ {:.1f} m".format(float(self.vol_r[ix])), **title_style)
        else:
            print(is_float(text))

    def get_wfid(self, input_az):
        angdiff = np.abs(self.fan_az-input_az)
        output_ind = np.where(angdiff == np.min(angdiff))[0]
        return output_ind[0]

    def get_vol_rid(self, input_r):
        dist = self.get_dist()
        rdiff = np.abs(dist-input_r)
        output_ind = np.where(rdiff == np.min(rdiff))[0]
        return output_ind[0]

    def get_dist(self):
        bw = self.bw_slider.value() * 1e6
        slope = bw / ramp_time_s
        dist = (self.freq - signal_freq) * c / (4 * slope)
        return dist

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

    def set_range_res(self):
        """ Sets the RF bandwidth
        Returns:
            None
        """
        bw = self.bw_slider.value() * 1e6
        slope = bw / ramp_time_s
        dist = (self.freq - signal_freq) * c / (4 * slope)
        print("New slope: %0.2fMHz/s" % (slope / 1e6))

        my_phaser.freq_dev_range = int(bw / 4)  # frequency deviation range in Hz
        my_phaser.freq_dev_step = int(
            bw / num_steps/ 4
        )
        my_phaser.enable = 0
        self.change_x_axis(self.x_axis_check.isChecked())
        self.clear_img()

    def change_x_axis(self, state):
        """ Toggles between showing frequency and range for the x-axis
        Args:
            state (QtCore.Qt.Checked) : State of check box
        Returns:
            None
        """
        dist = self.get_dist()
        if self.x_axis_check.isChecked() == True:
            print("Range axis")
            # range_x = np.max(dist)
            self.plot_xaxis = dist
            self.fft_plot.setTitle("Received Signal - Range", **title_style)
            self.fft_plot.setLabel("bottom", text="Range", units="m")
            # self.fft_plot.setXRange(0, range_x/2)
            self.fft_plot.setXRange(0, 5)
            for ip in range(3):
                # self.waterfall[ip].setRange(yRange=(0, range_x/2))
                self.waterfall[ip].setRange(yRange=(0, 5))
                self.waterfall[ip].setLabel("left", "Range", units="m")
                self.set_Quads(self.imageitem[ip])
            self.set_Quads(self.fanimage, plot_az=True)
            self.fanaxs.setRange(yRange=(0, 5))
            self.fanaxs.setLabel("left", "Range", units="m")
        else:
            print("Frequency axis")
            self.plot_xaxis = self.freq
            # self.fft_plot.setXRange(0, 200e3)
            self.fft_plot.setXRange(self.freq[0],self.freq[-1]/2)
            self.fft_plot.setTitle("Received Signal - Frequency Spectrum", **title_style)
            self.fft_plot.setLabel("bottom", text="Frequency", units="Hz")
            for ip in range(3):
                self.waterfall[ip].setRange(yRange=(self.freq[0],self.freq[-1]/2))
                self.waterfall[ip].setLabel("left", "Frequency", units="Hz")
                self.set_Quads(self.imageitem[ip])
            self.set_Quads(self.fanimage, plot_az=True)
            self.fanaxs.setRange(yRange=(self.freq[0],self.freq[-1]/2))
            self.fanaxs.setLabel("left", "Frequency", units="Hz")

    def range_correction(self, state):
        if state == QtCore.Qt.Checked:
            self.r_cal = np.full((self.freq.size),-999)
            dist = self.get_dist()
            self.r_cal[dist>0] = 20*np.log10(dist[dist>0]) - 20*np.log10(5)
            self.r_cal = self.r_cal[np.newaxis,:]
        else:
            self.r_cal = np.zeros((self.freq.size))[np.newaxis,:]

    def clear_img(self):
        self.img = np.full((3, self.tframe.size, self.freq.size),-300)

    def reset_fa(self):
        self.fabuf = np.full((*self.thetax.shape, 2), 0, dtype=np.complex_)
        self.cur_ori = 0
        self.vol_int = 0
        newt = time.time()
        self.cur_time = newt
        print(self.cur_ori)

    def reset_display_range(self):
        for ip in range(2):
            self.volplot[ip].setRange(xRange=(self.thetay[0,0], self.thetay[-1,0] ), yRange=(self.thetax[0,0],self.thetax[0,-1]))
        self.fft_plot.setYRange(-55, 5)
        if self.x_axis_check.isChecked() == True:
            for ip in range(3):
                self.waterfall[ip].setRange(xRange=(self.tframe[0], self.tframe[-1] ), yRange=(0, 5))
            self.fanaxs.setRange(xRange=(self.fan_az[0], self.fan_az[-1] ), yRange=(0, 5))
            self.fft_plot.setXRange(0, 5)
        else:
            for ip in range(3):
                self.waterfall[ip].setRange(xRange=(self.tframe[0], self.tframe[-1] ), yRange=(self.freq[0],self.freq[-1]/2))
            self.fanaxs.setRange(xRange=(self.fan_az[0], self.fan_az[-1] ), yRange=(self.freq[0],self.freq[-1]/2))
            self.fft_plot.setXRange(self.freq[0],self.freq[-1]/2)

    def cleanup(self):
        # Release resources here
        my_sdr.tx_destroy_buffer()
        print("Tx Buffer Destroyed!")

        # # To disable TDD and revert to non-TDD (standard) mode
        tdd.enable = False
        sdr_pins.gpio_phaser_enable = False
        tdd.out_channel1_polarity = not(sdr_pins.gpio_phaser_enable)
        tdd.out_channel2_polarity = sdr_pins.gpio_phaser_enable
        tdd.enable = True
        tdd.enable = False
        print("Disable TDD and revert to non-TDD (standard) mode")

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
App.aboutToQuit.connect(win.cleanup)
index = 0

ref = 2 ** 12
start_time = time.time()
def update():
    frdata = np.zeros((win.freq.size),dtype=np.complex_)
    fxdata = np.zeros((win.fan_az.size, win.freq.size),dtype=np.complex_)
    fadata = np.zeros((*win.thetax.shape, 2),dtype=np.complex_)
    # updatebuf = np.zeros((win.az.size,*win.thetax.shape, 2),dtype=np.complex_)
    ath=np.abs(win.az[-1]-win.az[-2])/1.9
    rx_bursts = np.zeros((CPI_pulse, good_ramp_samples), dtype=np.complex_)
    # win_funct = np.blackman(win.freq.size)
    win_funct = nuttall_window(good_ramp_samples)
    for ia, steer in enumerate(win.az):
        my_phaser.set_beam_phase_diff(steer_angle_to_phase_diff(steer, output_freq,0.014)*180/np.pi)
        my_phaser.gpios.gpio_burst = 0
        my_phaser.gpios.gpio_burst = 1
        my_phaser.gpios.gpio_burst = 0
        data = my_sdr.rx()
        win.update_ori()

        element_ph = win.DBF(steer)
        data_ar = np.array(data)
        data_ar = 1 / win.fft_size * np.fft.fft( data_ar[:,start_offset_samples:start_offset_samples+good_ramp_samples] * win_funct, n=win.fft_size)
        fainc = np.sum(data_ar[:,np.newaxis,[win.vol_ind]]*np.exp(-1j*element_ph[:,:,:,np.newaxis]),axis=0)
        update_pixel = np.abs(element_ph[0,:,:])<np.deg2rad(ath)
        fadata[update_pixel,:]=fainc[update_pixel,:]

        element_ph = win.DBF1d(steer)
        fainc = np.sum(data_ar[:,np.newaxis,:]*np.exp(-1j*element_ph[:,:,np.newaxis]),axis=0)
        update_pixel = np.abs(win.fan_az-steer)<ath
        fxdata[update_pixel,:]=fainc[update_pixel,:]
        frdata += data_ar[0]+data_ar[1]
        # updatebuf[ia,:] = fadata

    win.fabuf += fadata
    win.vol_int += 1
    # if win.vol_int%2 ==0 :
    #     with open('test{:>02d}.npy'.format(win.vol_int), 'wb') as f:
    #         np.save(f, updatebuf)
    win.integral_num.setText("{:d} scans integrated.".format(win.vol_int))
    win.ori_dis.setText("current orientation {:.3f} <html><sup>o</sup></html>".format(win.cur_ori))
    win.img = np.roll( win.img, 1, axis=1 )
    ampl = np.abs(fxdata)
    ampl = 20 * np.log10(ampl / ref + 10 ** -20)

    win.fan = ampl

    pr = np.abs(frdata)
    pr = 20 * np.log10(pr / ref + 10 ** -20)
    pr = pr - np.max(pr)
    # if np.abs(win.offset-np.max(win.fan))>5:
    #     win.offset=np.max(win.fan)
    for ip in range(3):
        win.img[ip,0,:] = ampl[win.wfid[ip],:]
        win.imageitem[ip].setImage(win.img[ip,:,:] - win.offset + win.r_cal, autoLevels=False)
    win.fanimage.setImage(win.fan - win.offset + win.r_cal, autoLevels=False)
    volcut = win.fabuf
    volcut = np.abs(volcut)
    volcut = 20 * np.log10(volcut / ref + 10 ** -20)
    win.volitem[0].setImage(volcut[:,:,0] - win.offset + win.r_cal[0,win.vol_ind[0]] - 15, autoLevels=False)
    win.volitem[1].setImage(volcut[:,:,1] - win.offset + win.r_cal[0,win.vol_ind[1]] - 15, autoLevels=False)
    # with open('fig{:>02d}.npy'.format(win.vol_int), 'wb') as f:
    #     np.save(f, volcut[:,:,0])
    win.fft_curve.setData(win.plot_xaxis, pr)
    global start_time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time*1e3} ms")
    start_time = end_time

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

# start the app
sys.exit(App.exec())

# Clean up / close connections
del my_sdr
del my_phaser
