import time
from qcodes.instrument_drivers.rohde_schwarz import RohdeSchwarzSGS100A

#Edit these parameters:
Frequency = 180e6
Power = -5
IP = '169.254.190.173'



sgsa = RohdeSchwarzSGS100A("SGSA100", "TCPIP0::"+IP+"::inst0::INSTR")#insert the IP address in between the :: ::
sgsa.print_readable_snapshot(update=True)

# set a power and a frequency

sgsa.frequency(Frequency)   #Input frequency here! (between 1MHz to 6GHz)
sgsa.power(Power)          #Input a power (between -20 to +25 dBm)

# start RF output
sgsa.status(True)

print('running\n ----------------------------------------------')
sgsa.print_readable_snapshot(update=True)





#
# time.sleep(60)#how long the device will run for.
#
# #stop RF outout
# sgsa.status(False)
# print('stopped')
# sgsa.print_readable_snapshot(update=True)










