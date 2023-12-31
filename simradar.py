import time
import threading
import numpy as np
from scipy import constants 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import queue

def fftconvolve(ax,ay,axis=-1):
    nex=ax.shape[axis]+ay.shape[axis]-1
    return np.fft.ifft(np.fft.fft(ax,n=nex,axis=axis)*np.fft.fft(ay,n=nex,axis=axis),axis=axis)

def DBF(gaz,gele,wl,a0,element_pos,we):
    k=np.pi*2/wl
    ar = np.array([np.sin(gele)*np.sin(gaz), np.sin(gele)*np.cos(gaz), np.cos(gele)])
    ar = ar-a0
    Exy=np.zeros((*we.shape[:2],*ar.shape[1:]))
    for ie in range(element_pos.shape[1]):
        de = element_pos[:,ie,:]
        element_phase = np.sum(ar[np.newaxis,:,:,:]*de[:,:,np.newaxis,np.newaxis],axis=1)
        xe = we[:,:,ie]
        Exy = Exy + xe[:,:,np.newaxis,np.newaxis]*np.exp(1j*k*element_phase)[:,np.newaxis,:,:]
    return Exy

def element_rotate(element_pos,orientation):
    # this rotation is clockwise not math degree
    out_pos = np.zeros((orientation.size,*element_pos.shape))
    for ik, displace in enumerate(orientation):
        deltadeg = displace*np.pi/180
        Mr = np.array([[np.cos(deltadeg),np.sin(deltadeg),0],[-np.sin(deltadeg),np.cos(deltadeg),0],[0,0,1]])
        out_pos[ik,:,:] = np.transpose(np.matmul(Mr,np.transpose(element_pos)))
    return out_pos

class RadarHost:
    def __init__(self,queue_fig):
        # self.state = "initial"
        self.lock = threading.Lock()
        self.on = True
        self.queue_fig = queue_fig
        self.TS = 290        # noise temperature K
        self.pulse_interval = 1 # intergal pulses inveral (trick for saving resources in simulation)
        self.npulse = 512    # number of pulses in sequence
        self.Pt = 100        # peak power    W
        self.G = 100         # Gain
        self.F0 = 10.35e9       # carrier frequency Hz
        self.wavelength = constants.c/self.F0     # radar wavelength  m
        self.B = 20e6
        self.PRF = 300.e3     # PRF   Hz
        self.set_transceiver_prf(self.PRF)
        self.platform_dps = 0
        self.fire_secs = 3
        de = self.wavelength/2
        xx,yy = np.meshgrid(np.arange(-3.5,3.6,1),np.arange(-1.5,1.6,1))
        self.element_loc0 = np.array([xx.flatten(),yy.flatten(),np.zeros(xx.size)]).transpose()*de
        self.cur_ori = 0
        theta0 = 0 * np.pi/180  # azimuth y->x
        phi0 = 0 * np.pi/180    #zenith
        self.a0 = np.array([np.sin(phi0)*np.sin(theta0), np.sin(phi0)*np.cos(theta0), np.cos(phi0)])[:,np.newaxis,np.newaxis]
        self.set_default_target()
        self.set_default_beams()
        self.exit = False
        self.capture_fig = False
        self.figN = 0
        self.ptitle = ''
        self.initdraw = True
        self.algorithm = self.subarray

    def handle_axes_cmd(self, cmds):
        if cmds[1]=='r':
            self.set_axes_DBF_r(int(cmds[2]))

    def set_axes_DBF_r(self, inpara):
        self.DBF_r = inpara

    def handle_capture_cmd(self, cmds):
        if self.capture_fig:
            with self.lock:
                self.capture_fig = False
                print('Turn off figure capture.')
                plt.close(self.fig)
                del self.fig
                del self.pcm
        else:
            with self.lock:
                print('Turn on figure capture.')
                self.init_fig()
                self.figN = 0

    def init_fig(self):
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        divider = make_axes_locatable(self.ax)
        ax_cb = divider.new_horizontal(size="5%", pad=0.1)
        self.cax = self.fig.add_axes(ax_cb)
        self.capture_fig = True
        os.makedirs('./figshot/', exist_ok=True)
        self.initdraw = True

    def flush_figure(self):
        if self.capture_fig:
            self.queue_fig.put((self.fig,f'./figshot/{self.figN:0>4d}.png'))
            # self.fig.savefig(f'./figshot/{self.figN:0>4d}.png')
            self.figN += 1
            # self.pulse_interval = self.PRF//self.npulse//36*self.figN

    def handle_mission_cmd(self, cmds):
        if int(cmds[1]) == 0:
            self.set_default_target()
            self.platform_dps = 0
            self.cur_ori = 0
        elif int(cmds[1]) == 1:
            Rtar_list = [ 50, 50, 50]        # target range
            Atar_list = [ 0, 90, 45]              # target azimuth
            Etar_list = [ 0, 30, 45]              # target elevation
            Vtar_list = [0, 0, 0]               # fix target radial velocity
            rcs_list = [2, 5, 10]         # radar cross section   m^2
            self.target_para = [Rtar_list, Vtar_list, rcs_list, Atar_list, Etar_list]
            self.DBF_r = Rtar_list[0]
            self.platform_dps = 185
            self.pulse_interval = self.PRF//self.npulse
        elif int(cmds[1]) == 2:
            Rtar_list = [ 50, 50, 50]        # target range
            Atar_list = [ 0, 90, 45]              # target azimuth
            Etar_list = [ 0, 30, 45]              # target elevation
            Vtar_list = [60, -40, 0]               # fix target radial velocity
            rcs_list = [2, 5, 10]         # radar cross section   m^2
            self.target_para = [Rtar_list, Vtar_list, rcs_list, Atar_list, Etar_list]
            self.DBF_r = Rtar_list[0]
            self.platform_dps = 185
            self.pulse_interval = self.PRF//self.npulse
        elif int(cmds[1]) == 3:
            Rtar_list = [ 30, 60, 60]        # target range
            Atar_list = [ 45,225,315]              # target azimuth
            Etar_list = [ 20,15,25]              # target elevation
            Vtar_list = [ 0,0,0]               # fix target radial velocity
            rcs_list =  [10,6,10]         # radar cross section   m^2
            self.target_para = [Rtar_list, Vtar_list, rcs_list, Atar_list, Etar_list]
            self.DBF_r = Rtar_list[1]
            self.fire_secs = 10
            self.platform_dps = 360+10/self.fire_secs
            self.pulse_interval = self.PRF//self.npulse
        elif int(cmds[1]) == 4:
            Rtar_list = [ 30, 60, 60]        # target range
            Atar_list = [ 45,225,315]              # target azimuth
            Etar_list = [ 20,15,25]              # target elevation
            Vtar_list = [ 0,0,0]               # fix target radial velocity
            rcs_list =  [10,6,10]         # radar cross section   m^2
            self.target_para = [Rtar_list, Vtar_list, rcs_list, Atar_list, Etar_list]
            self.DBF_r = Rtar_list[1]
            self.fire_secs = 10
            self.platform_dps = 360+10/self.fire_secs
            self.pulse_interval = 1
        elif int(cmds[1]) == 5:
            Rtar_list = [ 30, 60, 60]        # target range
            Atar_list = [ 45,225,315]              # target azimuth
            Etar_list = [ 20,15,25]              # target elevation
            Vtar_list = [ 0,0,0]               # fix target radial velocity
            rcs_list =  [10,6,10]         # radar cross section   m^2
            self.target_para = [Rtar_list, Vtar_list, rcs_list, Atar_list, Etar_list]
            self.DBF_r = Rtar_list[1]
            self.fire_secs = 3
            self.platform_dps = 360
            self.pulse_interval = self.PRF//self.npulse//36
        else:
            print("invalid mission command.")

    def set_default_beams(self):
        beamx = np.arange(-45,46,1.5) * np.pi/180
        beamy = np.arange(-45,46,1.5) * np.pi/180
        # beamx = np.arange(-30,31,1) * np.pi/180
        # beamy = np.arange(-30,31,1) * np.pi/180
        thetax, thetay = np.meshgrid(beamx, beamy)
        self.thetaele = np.arccos(np.sqrt(1-np.sin(thetax)**2-np.sin(thetay)**2))
        self.thetaaz = np.arctan2(np.sin(thetax),np.sin(thetay))
        self.thetax = thetax*180/np.pi
        self.thetay = thetay*180/np.pi

    def add_new_target(self, inpara):
        nt = (map(float,inpara.split(',')))
        for var, tp in zip(nt,self.target_para):
            tp.append(var)

    def clear_target(self):
        self.target_para = [[],[],[],[],[]]

    def handle_object_cmd(self,cmds):
        if cmds[1]=='a':
            self.add_new_target(cmds[2])
        elif cmds[1]=='c':
            self.clear_target()
        else:
            print("invalid object command.")

    def set_default_target(self):
        Rtar_list = [ 50, 50, 50]        # target range
        Atar_list = [ 0, 90, 45]              # target azimuth
        Etar_list = [ 0, 30, 45]              # target elevation
        Vtar_list = [60, -40, 0]               # fix target radial velocity
        rcs_list = [2, 5, 10]         # radar cross section   m^2
        self.target_para = [Rtar_list, Vtar_list, rcs_list, Atar_list, Etar_list]
        self.DBF_r = Rtar_list[0]
        self.pulse_interval = 1

    def set_udchrip_waveform(self):
        txp = np.arange(0,self.Tx,self.ts)    # discrete time of pulse
        wd = np.ones(txp.size)
        tx = np.zeros(txp.shape,dtype=np.complex128)
        tx[:tx.size//2] = self.Pt*wd[:tx.size//2]*np.exp((-np.pi*self.B*txp[:tx.size//2]+2*np.pi*self.chirp*txp[:tx.size//2]**2)*1j) # transmit waveform
        txpl=txp[tx.size//2:]-self.Tx/2
        tx[tx.size//2:] = self.Pt*wd[tx.size//2:]*np.exp((np.pi*self.B*txpl-2*np.pi*self.chirp*txpl**2)*1j)
        self.tx = tx
        self.txp = txp
        tf = self.txp
        Faxe=np.arange(-1/2/self.ts,1/2/self.ts,1/self.ts/tf.size)  # frequency after fft
        self.Faxe = Faxe[np.newaxis,:]

    def set_transceiver_npulse(self, inpara):
        self.npulse = inpara

    def set_transceiver_pulse_interval(self, inpara):
        self.pulse_interval = inpara

    def set_transceiver_prf(self, inpara):
        self.PRF = inpara
        self.PRT = 1/self.PRF
        self.Tx = 1/self.PRF      # pulse width
        self.Va = constants.c*self.PRF/4/self.F0
        self.chirp = self.B/self.Tx
        self.ts = self.PRF/(2*(2*self.chirp))       # sample rate   s
        self.nfft = int(2**(np.log(1/self.PRF/self.ts)//np.log(2)+2))      # zero padding in range fft
        self.Lx = self.Tx/self.ts
        self.Ra = constants.c/self.PRF/2
        self.set_udchrip_waveform()

    def handle_transceiver_cmd(self,cmds):
        if cmds[1]=='n':
            self.set_transceiver_npulse(int(cmds[2]))
        elif cmds[1]=='prf':
            self.set_transceiver_prf(float(cmds[2]))
        elif cmds[1]=='pi':
            self.set_transceiver_pulse_interval(float(cmds[2]))
        else:
            print("invalid transceiver command.")

    def set_platform_orientation(self, inpara):
        self.cur_ori = inpara

    def set_platform_speed(self, inpara):
        self.platform_dps = inpara

    def update_platform_orientation(self):
        self.element_pos = element_rotate(self.element_loc0, self.cur_ori + self.platform_dps*np.arange(self.npulse)*self.PRT*self.pulse_interval)
        self.cur_ori = self.cur_ori + self.platform_dps*self.fire_secs
        self.cur_ori = self.cur_ori % 360

    def handle_platform_cmd(self,cmds):
        if cmds[1]=='o':
            self.set_platform_orientation(float(cmds[2]))
        elif cmds[1]=='s':
            self.set_platform_speed(float(cmds[2]))
        else:
            print("invalid platform command.")

    def handle_summary(self,cmds):
        if len(cmds)==1:
            print("\nplatform")
            self.summary_platform()
            print("\ntransceiver")
            self.summary_transceiver()
            print("\nobjects")
            self.summary_object()
        elif cmds[1]=='p':
            self.summary_platform()
        elif cmds[1]=='t':
            self.summary_transceiver()

    def summary_platform(self):
        print(f'\t current orientation \t { self.cur_ori % 360 :.1f} degree')
        print(f'\t platform speed \t { self.platform_dps :.1f} dps')

    def summary_transceiver(self):
        print(f'\t n pulse / fire \t { self.npulse :d} ')
        print(f'\t pulse interval \t { int(self.pulse_interval) :d} ')
        print(f'\t PRF \t\t\t { self.PRF/1e3 :.1f} kHz')
        print(f'\t F0 \t\t\t { self.F0/1e9 :.1f} GHz')

    def summary_object(self):
        for l,v in zip(['Range','Vr','RCS','Azimuth','Zenith'],self.target_para):
            print('\t'+l+'\t\t\t',v)

    def fire_pulse(self):
        element_loct = self.element_pos
        target_p = self.target_para
        tx = self.tx
        tm = np.arange(self.npulse)*self.PRT*self.pulse_interval
        tm = tm[:,np.newaxis]
        tf = self.txp
        fast_time = tf[np.newaxis,:]
        total_rx = np.zeros((self.npulse,tf.size,int(element_loct.shape[1])),dtype=tx.dtype)
        for Rtar,Vtar,rcs, azt, elt in zip(*target_p):
            tau_0 = 2*Rtar/constants.c # first pulse target time delay
            FD = -2*Vtar/self.wavelength  # target doppler
            ARtar = constants.c*(tau_0-np.floor(tau_0/self.PRT)*self.PRT)/2 # apperent range of target
            Radar_c = np.sqrt(self.G**2*self.wavelength**2*rcs/(4*np.pi)**3)
            # *np.exp(2j*np.pi*np.random.random(1)) # radar constant without scale of range
            rr = constants.c*(tau_0)/2       # range
            Radar_c = Radar_c/rr**2     # return power
        
            rx=np.zeros(tf.shape,dtype=tx.dtype)       # received signal
            rx[:tx.size]=tx
        
            tau_m = 2*(Rtar+tm*Vtar)/constants.c
            rx=np.tile(rx,(self.npulse,1))
            Radar_c=Radar_c
            frx=np.fft.fftshift(np.fft.fft(rx,axis=1),axes=1) 
                # frequency content of received signal
            dfrx=frx*np.exp(-2j*np.pi*self.Faxe*tau_m)   # time delay in frequency domain
        
            rx=np.fft.ifft(np.fft.fftshift(dfrx,axes=1))*Radar_c
            rx=rx*np.exp(2j*np.pi*FD*fast_time)*np.exp(-2j*np.pi*self.F0*tau_m)
            az = azt*np.pi/180
            ele = elt*np.pi/180
            targetRvector = np.array([np.sin(ele)*np.sin(az),np.sin(ele)*np.cos(az),np.cos(ele)])[np.newaxis,np.newaxis,:]
            dl = np.sum(element_loct*targetRvector,axis=2)[:,np.newaxis,:]
            total_rx=total_rx+rx[:,:,np.newaxis]*np.exp(-2j*np.pi*dl/self.wavelength)
        self.rx_buf = total_rx
        if self.capture_fig:
            self.algorithm()
            self.flush_figure()


    def beamforming(self,element_loct,rx, bftype ='All Digital'):
        # if hasattr(self, 'fig'):
        tx = self.tx
        crx = rx[:,:tx.shape[0]//2,:]*np.conj(tx[np.newaxis,:tx.shape[0]//2,np.newaxis])
        Fr = np.arange(-1/2/self.ts,1/2/self.ts,1/self.ts/self.nfft)
        r = -Fr/(2*self.chirp)*constants.c/2
        crx = np.fft.fftshift(np.fft.fft(crx,n = self.nfft,axis=1),axes=1)
        clasper = np.where(np.abs(r-self.DBF_r)==np.min(np.abs(r-self.DBF_r)))[0][0]
        Eloc = DBF(self.thetaaz,self.thetaele,self.wavelength,self.a0,element_loct,crx[:,clasper:clasper+2,:])
        Beam = np.mean(np.abs(Eloc)**2,axis=0)
        self.plot_data = Beam[0,:,:]
        pt = bftype + f" R = {self.DBF_r:d} m "+r'$\alpha$'+f" = {int(self.cur_ori):d}"+r'$^o$'
        self.ptitle = pt + r'$,\Delta \alpha$'+f" = {int(self.platform_dps*self.npulse*self.PRT*self.pulse_interval):d}"+r'$^o$'
        if self.initdraw:
            self.xlabel = r'$\theta_x$'
            self.ylabel = r'$\theta_y$'
            self.xx = np.squeeze(self.thetax)
            self.yy = np.squeeze(self.thetay)
            self.pcm = self.init_pcm(self.fig, self.ax, self.cax)
        else:
            self.update_pcm()

        # self.flush_figure()

    def all_digital(self):
        self.beamforming(self.element_pos,self.rx_buf, bftype ='All Digital')

    def subarray(self):
        element_loct = self.element_pos
        rx = self.rx_buf
        subarray_rx = np.sum(rx.reshape((rx.shape[0],rx.shape[1],4,8)), axis=2)
        subarray_loc0 = np.mean(element_loct.reshape((element_loct.shape[0],4,8,3)), axis=1)
        self.beamforming(subarray_loc0,subarray_rx, bftype ='Subarray')

    def update_pcm(self):
        self.pcm.set_array(self.plot_data.ravel())
        self.ax.set_title(self.ptitle)

    # def update_dpcm(self):
    #     self.dpcm.set_array(self.plot_data.ravel())
    #     self.dpcm.axes.set_title(self.ptitle)
    def update_dpcm(self,dummy=None):
        self.update_pcm()
        # self.pcm.set_array(self.plot_data.ravel())
        # self.ax.set_title(self.ptitle)
        if not self.queue_fig.empty():
            fig, filename = self.queue_fig.get()
            output_plot(fig, filename)

    def range_doppler_map(self):
        rx = self.rx_buf
        tx = self.tx
        Fr = np.arange(-1/2/self.ts,1/2/self.ts,1/self.ts/self.nfft)
        r = -Fr/(2*self.chirp)*constants.c/2
        crxu = np.mean(rx[:,:tx.shape[0]//2,:],axis=2)*np.conj(tx[np.newaxis,:tx.shape[0]//2])
        nv=int(2**(np.log(crxu.shape[0])//np.log(2)+2))
        rdm=np.abs(np.fft.fftshift(np.fft.fft(np.fft.fftshift(np.fft.fft(crxu,n=nv,axis=0),axes=0),n=self.nfft,axis=1),axes=1))
        Fv=np.arange(-self.PRF/2,self.PRF/2,self.PRF/nv)
        # rdm = 10*np.log10(rdm)
        self.Vaxes = -Fv/2*self.wavelength
        self.raxes = r
        self.plot_data = np.transpose(rdm)
        self.ptitle = f" Phaser orientation = {int(self.cur_ori):d}"+r'$^o$'
        if self.initdraw:
            self.xlabel = 'Doppler velocity (m/s)'
            self.ylabel = 'Range (m)'
            self.xx = self.Vaxes
            self.yy = self.raxes
            self.pcm = self.init_pcm(self.fig, self.ax, self.cax)
            self.ax.set_xlim(-100,100)
            self.ax.set_ylim(0,200)
        else:
            self.update_pcm()
        # self.flush_figure()

    def init_pcm(self, fig, ax, cax):
        pcm = ax.pcolormesh(self.xx, self.yy, self.plot_data)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_title(self.ptitle)
        fig.colorbar(pcm, cax = cax)
        self.initdraw = False
        return pcm

    # def init_rdm(self,title):
    #     pcm = self.ax.pcolormesh(self.Vaxes,self.raxes,self.plot_data)
    #     self.ax.set_xlim(-100,100)
    #     self.ax.set_ylim(0,200)
    #     self.ax.set_xlabel('Doppler velocity (m/s)')
    #     self.ax.set_ylabel('Range (m)')
    #     self.fig.colorbar(pcm, ax=self.ax)
    #     self.pcm = pcm
    #     self.ax.set_title(title)
    #     self.initdraw = False

    def handle_display_cmd(self,cmds):

        if cmds[1]=='a':
            self.algorithm = self.all_digital
            # self.all_digital()
            # animation = FuncAnimation(fig, self.all_digital, frames=1, interval=self.fire_secs*1e3)
        elif cmds[1]=='s':
            self.algorithm = self.subarray
            # self.subarray()
            # animation = FuncAnimation(fig, self.subarray, frames=1, interval=self.fire_secs*1e3)
        elif cmds[1]=='r':
            self.algorithm = self.range_doppler_map
            # self.range_doppler_map()
            # animation = FuncAnimation(fig, self.range_doppler_map, frames=1, interval=self.fire_secs*1e3)
        else:
            print("invalid beamforming command.")
        if self.initdraw:
            self.init_fig()
        else:
            with self.lock:
                plt.close(self.fig)
                del self.fig
                del self.pcm
                self.init_fig()
        self.initdraw = True
        self.algorithm()
        # fig = self.fig
        # dfig, dax = plt.subplots()
        # # self.fig = fig
        # # self.ax = ax
        # divider = make_axes_locatable(dax)
        # ax_cb = divider.new_horizontal(size="5%", pad=0.1)
        # dcax = dfig.add_axes(ax_cb)
        # self.dpcm = self.init_pcm(dfig, dax, dcax)
        animation = FuncAnimation(self.fig, self.update_dpcm, frames=1, interval=self.fire_secs*1e3)
        plt.show()
        with self.lock:
            plt.close(self.fig)
            del self.fig
            del self.pcm
            self.init_fig()

    def run(self):
        while not (self.exit):
            if self.on:
                with self.lock:
                    self.update_platform_orientation()
                    tic = time.time()
                    self.fire_pulse()
                    elapsed_time = time.time() - tic
                    print(f"Fire {self.npulse:d} pulses CPU Elapsed Time: {elapsed_time:.2f} seconds")
            time.sleep(3)

    def wait(self):
        with self.lock:
            self.on = False

    def resume(self):
        with self.lock:
            self.on = True

    def pause(self,secs):
        with self.lock:
            for ik in range(secs):
                print("pause",ik)
                time.sleep(1)

def output_plot(fig,figname):
    fig.savefig(figname)

def command_host(qc):
    while 1:
        cmdbuf = input('Command Center await.\n')
        cmd = ''
        for c in cmdbuf:
            if c=='\b':
                cmd=cmd[:-1]
            else:
                cmd+=c
        qc.put(cmd)
        cmd = ' '.join(cmd.split())
        cmds = cmd.split()
        if cmds[0]=='q':
            break

def command_handle(radarobj, cmd):
    if cmd=='':
        return
    def response(radarobj, cmd):
        max_retries = 15
        retries = 0
        cmd = ' '.join(cmd.split())
        cmds = cmd.split()
        while retries < max_retries:
            if not radarobj.lock.locked():
                if cmds[0]=='p':
                    radarobj.handle_platform_cmd(cmds)
                elif cmds[0]=='a':
                    radarobj.handle_axes_cmd(cmds)
                elif cmds[0]=='d':
                    radarobj.handle_display_cmd(cmds)
                elif cmds[0]=='t':
                    radarobj.handle_transceiver_cmd(cmds)
                elif cmds[0]=='s':
                    radarobj.handle_summary(cmds)
                elif cmds[0]=='m':
                    radarobj.handle_mission_cmd(cmds)
                elif cmds[0]=='o':
                    radarobj.handle_object_cmd(cmds)
                elif cmds[0]=='f':
                    radarobj.handle_capture_cmd(cmds)
                elif cmds[0]=='q':
                    radarobj.exit = True
                    exit()
                elif cmds[0]=='w':
                    radarobj.wait()
                    print("radar core standby.")
                elif cmds[0]=='r':
                    radarobj.resume()
                    print("radar core resume.")
                else:
                    print("unknown command kind.")
                break
            else:
                print("Thread is locked. Retrying...")
                retries += 1
                time.sleep(0.2)  # Optional: Add a delay between retries
                if retries == max_retries:
                    print("Time out. Command skipped. ")

    if cmd[0]=='l':
        radarobj.lock.acquire()
    elif cmd[0]=='g':
        radarobj.lock.release()
    else:
        response(radarobj,cmd)

if __name__ == "__main__":
    queue_fig = queue.Queue()
    queue_cmd = queue.Queue()
    radarcore = RadarHost(queue_fig)
    thread = threading.Thread(target=radarcore.run)
    input_host = threading.Thread(target=command_host, args=(queue_cmd,))
    thread.start()
    input_host.start()

    while 1:
        if not queue_cmd.empty():
            cmd = queue_cmd.get()
            command_handle(radarcore,cmd)
        if not queue_fig.empty():
            fig, filename = queue_fig.get()
            output_plot(fig, filename)
        time.sleep(0.2)

    thread.join()