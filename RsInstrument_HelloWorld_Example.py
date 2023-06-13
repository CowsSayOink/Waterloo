# Simple example on how to use the RsInstrument module for remote-controlling yor VISA instrument
# Preconditions:
# - Installed RsInstrument Python module (see the attached RsInstrument_PythonModule folder Readme.txt)
# - Installed VISA e.g. R&S Visa 5.12.x or newer



#
# from RsInstrument.RsInstrument import RsInstrument
#
# resource_string_1 = 'TCPIP::169.254.190.173::INSTR'  # Standard LAN connection (also called VXI-11)
# resource_string_2 = 'TCPIP::169.254.190.173::hislip0'  # Hi-Speed LAN connection - see 1MA208
# resource_string_3 = 'GPIB::20::INSTR'  # GPIB Connection
# resource_string_4 = 'USB::0x0AAD::0x0119::022019943::INSTR'  # USB-TMC (Test and Measurement Class)
# resource_string_5 = 'RSNRP::0x0095::104015::INSTR'  # R&S Powersensor NRP-Z86
# instr = RsInstrument(resource_string_1, True, False)
#
# idn = instr.query_str('*IDN?')
# print(f"\nHello, I am: '{idn}'")
# print(f'RsInstrument driver version: {instr.driver_version}')
# print(f'Visa manufacturer: {instr.visa_manufacturer}')
# print(f'Instrument full name: {instr.full_instrument_model_name}')
# print(f'Instrument installed options: {",".join(instr.instrument_options)}')
#
# # Close the session
# instr.close()
#
#



#
# from RsInstrument import *
# # Force use of the Rs Visa. For e.g.: NI Visa, use the "SelectVisa='ni'"
# instr = RsInstrument('TCPIP::169.254.190.173::INSTR', True, True, "SelectVisa='rs'")
#
# idn = instr.query_str('*IDN?')
# print(f"\nHello, I am: '{idn}'")
# print(f"\nI am using the VISA from: {instr.visa_manufacturer}")
#
# instr.reset()
# print(instr.idn_string)
#
#
# # Close the session
# instr.close()
#






#
# from RsInstrument import *
#
# instr = RsInstrument('TCPIP::169.254.190.173::INSTR', True, True)
# instr.write_str('*RST')
# response = instr.query_str('*IDN?')
# print(response)
#
# # Close the session
# instr.close()


import time

import qcodes as qc
from qcodes.instrument_drivers.rohde_schwarz import RohdeSchwarzSGS100A
sgsa = RohdeSchwarzSGS100A("SGSA100", "TCPIP0::169.254.190.173::inst0::INSTR")#insert the IP address in between the :: ::
sgsa.print_readable_snapshot(update=True)

# set a power and a frequency
sgsa.frequency(180e6)#Input frequency here!
sgsa.power(-5)

# start RF output
sgsa.status(True)

print('running\n ----------------------------------------------')
sgsa.print_readable_snapshot(update=True)

time.sleep(60)

#stop RF outout
sgsa.status(False)
print('stopped')
sgsa.print_readable_snapshot(update=True)










