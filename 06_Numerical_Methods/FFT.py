import numpy as np



class FFTAnalysis:


    def __init__(
            self,
            signal,
            sample_rate=1
    ):

        """
        signal:
            时间序列数据

        sample_rate:
            采样频率
        """


        self.signal=np.array(
            signal
        )

        self.sample_rate=sample_rate



    def transform(self):

        """
        FFT变换
        """


        n=len(
            self.signal
        )


        fft_result=np.fft.fft(
            self.signal
        )


        freq=np.fft.fftfreq(
            n,
            1/self.sample_rate
        )


        amplitude=np.abs(
            fft_result
        )


        return freq, amplitude



    def dominant_frequency(self):

        """
        找主要频率
        """


        freq,amp=self.transform()


        index=np.argmax(
            amp[1:]
        )+1


        return freq[index]