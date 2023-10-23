import time
import threading
import numpy as np
from scipy import constants 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

def fftconvolve(ax,ay,axis=-1):
    nex=ax.shape[axis]+ay.shape[axis]-1
    return np.fft.ifft(np.fft.fft(ax,n=nex,axis=axis)*np.fft.fft(ay,n=nex,axis=axis),axis=axis)

def DBF(gaz,gele,wl,a0,element_pos,we):
    k=np.pi*2/wl
    # thetaxy = np.arctan2(np.tan(thetay),np.tan(thetax))
    # phixy = np.arctan(np.sqrt(np.tan(thetay)**2+np.tan(thetax)**2))
    ar = np.array([np.sin(gele)*np.sin(gaz), np.sin(gele)*np.cos(gaz), np.cos(gele)])
    ar = ar-a0
    # print(ar)
    Exy=np.zeros((*we.shape[:2],*ar.shape[1:]))
    for ie in range(element_pos.shape[1]):
        de = element_pos[:,ie,:]
        element_phase = np.sum(ar[np.newaxis,:,:,:]*de[:,:,np.newaxis,np.newaxis],axis=1)
        # print(element_phase*180/np.pi)|
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
    def __init__(self):
        self.state = "initial"
        self.lock = threading.Lock()
        self.on = True

        self.TS = 290        # noise temperature K
        self.npulse = 64    # number of pulses in sequence
        self.nfft = 1024      # zero padding in fft
        self.Pt = 100        # peak power    W
        self.G = 100         # Gain
        self.F0 = 10.35e9       # carrier frequency Hz
        self.wavelength = constants.c/self.F0     # radar wavelength  m
        self.B = 20e6
        self.ts = 2e-8/3       # sample rate   s
        self.PRF = 600.e3     # PRF   Hz
        self.PRT = 1/self.PRF
        self.Tx = 1/self.PRF      # pulse width
        self.Va = constants.c*self.PRF/4/self.F0
        self.Lx = self.Tx/self.ts
        self.chirp = self.B/self.Tx
        self.Ra = constants.c/self.PRF/2    # maximum unambiguous range
        self.platform_dps = 0
        self.fire_secs = 3

        de = self.wavelength/2
        xx,yy = np.meshgrid(np.arange(-3.5,3.6,1),np.arange(-1.5,1.6,1))
        self.element_loc0 = np.array([xx.flatten(),yy.flatten(),np.zeros(xx.size)]).transpose()*de
        self.cur_ori = 0
        theta0 = 0 * np.pi/180  # azimuth y->x
        phi0 = 0 * np.pi/180    #zenith
        self.a0 = np.array([np.sin(phi0)*np.sin(theta0), np.sin(phi0)*np.cos(theta0), np.cos(phi0)])[:,np.newaxis,np.newaxis]
        self.set_udchrip_waveform()
        self.set_default_target()
        self.set_default_beams()
        self.exit = False

    def set_default_target(self):
        Rtar_list = [ 30, 30, 30]        # target range
        Atar_list = [ 60, 90, 0]              # target azimuth
        Etar_list = [-20, 30, 0]              # target elevation
        Vtar_list = [60, -40, 0]               # fix target radial velocity
        rcs_list = [10, 10, 10]         # radar cross section   m^2
        self.target_para = [Rtar_list, Vtar_list, rcs_list, Atar_list, Etar_list]
        self.DBF_r = Rtar_list[0]

    def set_default_beams(self):
        beamx = np.arange(-45,46,1.5) * np.pi/180
        beamy = np.arange(-45,46,1.5) * np.pi/180
        thetax, thetay = np.meshgrid(beamx, beamy)
        self.thetaele = np.arccos(np.sqrt(1-np.sin(thetax)**2-np.sin(thetay)**2))
        self.thetaaz = np.arctan2(np.sin(thetax),np.sin(thetay))
        self.thetax = thetax*180/np.pi
        self.thetay = thetay*180/np.pi

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
        # fast_time = tf[np.newaxis,:]
        Faxe=np.arange(-1/2/self.ts,1/2/self.ts,1/self.ts/tf.size)  # frequency after fft
        self.Faxe = Faxe[np.newaxis,:]

    def set_transceiver_npulse(self, inpara):
        self.npulse = inpara

    def handle_transceiver_cmd(self,cmds):
        if cmds[1]=='n':
            self.set_transceiver_npulse(int(cmds[2]))
        else:
            print("invalid transceiver command.")

    def set_platform_orientation(self, inpara):
        self.cur_ori = inpara

    def set_platform_speed(self, inpara):
        self.platform_dps = inpara

    def update_platform_orientation(self):
        self.element_pos = element_rotate(self.element_loc0, self.cur_ori + self.platform_dps*np.arange(self.npulse)*self.PRT)
        self.cur_ori = self.cur_ori + self.platform_dps*self.fire_secs

    def handle_platform_cmd(self,cmds):
        if cmds[1]=='o':
            self.set_platform_orientation(float(cmds[2]))
        elif cmds[1]=='s':
            self.set_platform_speed(float(cmds[2]))
        else:
            print("invalid platform command.")

    def handle_summary(self,cmds):
        if len(cmds)==1:
            print("platform\n")
            self.summary_platform()
            print("transceiver\n")
            self.summary_transceiver()
        elif cmds[1]=='p':
            self.summary_platform()
        elif cmds[1]=='t':
            self.summary_transceiver()

    def summary_platform(self):
        print(f'\t current orientation \t { self.cur_ori % 360 :.1f} degree')
        print(f'\t platform speed \t { self.platform_dps :.1f} dps')

    def summary_transceiver(self):
        print(f'\t n pulse / fire \t { self.npulse :d} ')
        print(f'\t PRF \t\t { self.PRF/1e3 :.1f} kHz')
        print(f'\t F0 \t\t { self.F0/1e9 :.1f} GHz')

    def fire_pulse(self):
        element_loct = self.element_pos
        target_p = self.target_para
        tx = self.tx
        tm = np.arange(self.npulse)*self.PRT
        tm = tm[:,np.newaxis]
        tf = self.txp
        fast_time = tf[np.newaxis,:]
        total_rx = np.zeros((self.npulse,tf.size,int(element_loct.shape[1])),dtype=tx.dtype)
        for Rtar,Vtar,rcs, azt, elt in zip(*target_p):
            tau_0 = 2*Rtar/constants.c # first pulse target time delay
            FD = 2*Vtar/self.wavelength  # target doppler
            ARtar = constants.c*(tau_0-np.floor(tau_0/self.PRT)*self.PRT)/2 # apperent range of target
            Radar_c = np.sqrt(self.G**2*self.wavelength**2*rcs/(4*np.pi)**3)
            # *np.exp(2j*np.pi*np.random.random(1)) # radar constant without scale of range
            rr = constants.c*(tau_0)/2       # range
            Radar_c = Radar_c/rr**2     # return power
        
            rx=np.zeros(tf.shape,dtype=tx.dtype)       # received signal
            rx[:tx.size]=tx
        
            tau_m = 2*(Rtar-tm*Vtar)/constants.c
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


    def beamforming(self,element_loct,rx, bftype ='All Digital'):
        # element_loct = self.element_pos
        # rx = self.rx_buf
        tx = self.tx
        crx = rx[:,:tx.shape[0]//2,:]*np.conj(tx[np.newaxis,:tx.shape[0]//2,np.newaxis])
        Fr = np.arange(-1/2/self.ts,1/2/self.ts,1/self.ts/self.nfft)
        r = Fr/(2*self.chirp)*constants.c/2
        # Beam[np.abs(r-30)==np.min(np.abs(r-30)),:,:]
        # crx[:,clasper:clasper+2,:]
        crx = np.fft.fftshift(np.fft.fft(crx,n = self.nfft,axis=1),axes=1)
        clasper = np.where(np.abs(r-self.DBF_r)==np.min(np.abs(r-self.DBF_r)))[0][0]
        Eloc = DBF(self.thetaaz,self.thetaele,self.wavelength,self.a0,element_loct,crx[:,clasper:clasper+2,:])
        # Eloc = np.fft.fftshift(np.fft.fft(Eloc,n = self.nfft,axis=1),axes=1)
        Beam = np.mean(np.abs(Eloc)**2,axis=0)
        self.plot_data = Beam[0,:,:]
        if hasattr(self, 'pcm'):
            self.update_beam()
        else:
            self.init_beam(bftype)

    def all_digital(self,dummy=None):
        self.beamforming(self.element_pos,self.rx_buf, bftype ='All Digital')

    def subarray(self,dummy=None):
        element_loct = self.element_pos
        rx = self.rx_buf
        subarray_rx = np.sum(rx.reshape((rx.shape[0],rx.shape[1],4,8)), axis=2)
        subarray_loc0 = np.mean(element_loct.reshape((element_loct.shape[0],4,8,3)), axis=1)
        self.beamforming(subarray_loc0,subarray_rx, bftype ='Subarray')

    def init_beam(self, bftype):
        pcm = self.ax.pcolormesh(np.squeeze(self.thetax),np.squeeze(self.thetay),np.squeeze(self.plot_data))
        # plt.ylim(np.min(self.thetay),np.max(self.thetay))
        # plt.xlim(np.min(self.thetax),np.max(self.thetax))
        self.ax.set_xlabel(r'$\theta_x$')
        self.ax.set_ylabel(r'$\theta_y$')
        self.ax.set_title(bftype + f" R = {self.DBF_r:d} m")
        plt.colorbar(pcm)
        self.pcm = pcm

    def update_beam(self):
        self.pcm.set_array(self.plot_data.ravel())

    def handle_beamform_cmd(self,cmds):
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        if cmds[1]=='a':
            self.all_digital()
            animation = FuncAnimation(fig, self.all_digital, frames=1, interval=self.fire_secs*1e3)
        elif cmds[1]=='s':
            self.subarray()
            animation = FuncAnimation(fig, self.subarray, frames=1, interval=self.fire_secs*1e3)
        else:
            print("invalid beamforming command.")
        plt.show()
        plt.close(fig)
        del self.pcm

    def run(self):
        while not (self.exit):
            if self.on:
                with self.lock:
                    self.update_platform_orientation()
                    tic = time.time()
                    self.fire_pulse()
                    elapsed_time = time.time() - tic
                    print(f"Fire {self.npulse:d} pulses Elapsed Time: {elapsed_time:.2f} seconds")
                    # if self.state == "initial":
                    #     self.initial_state()
                    # elif self.state == "state1":
                    #     self.state1()
                    # elif self.state == "state2":
                    #     self.state2()
            time.sleep(self.fire_secs)

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

def command_handle(radarobj):
    cmd = input('Command Center await.\n')

    def response(radarobj, cmd):
        max_retries = 10
        retries = 0
        cmd = ' '.join(cmd.split())
        cmds = cmd.split()
        while retries < max_retries:
            if not radarobj.lock.locked():
                # modify_shared_resource()
                # break  # Operation succeeded, exit the loop
                if cmds[0]=='p':
                    # radarobj.pause(int(cmd[1:]))
                    radarobj.handle_platform_cmd(cmds)
                elif cmds[0]=='b':
                    radarobj.handle_beamform_cmd(cmds)
                elif cmds[0]=='t':
                    radarobj.handle_transceiver_cmd(cmds)
                elif cmds[0]=='s':
                    radarobj.handle_summary(cmds)
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

    if cmd[0]=='l':
        radarobj.lock.acquire()
    elif cmd[0]=='g':
        radarobj.lock.release()
    else:
        response(radarobj,cmd)

        #     break
        # except threading.ThreadError:
            
            
    

if __name__ == "__main__":
    radarcore = RadarHost()
    # Create a thread for the state machine
    thread = threading.Thread(target=radarcore.run)

    # Start the thread
    thread.start()
    # time.sleep(5)
    # print('try to turn off machine.')
    # radarcore.close()
    while 1:
        command_handle(radarcore)
    # with radarcore.lock:
    #     radarcore.on = False
    # Wait for the thread to finish (in this case, it won't finish)
    thread.join()