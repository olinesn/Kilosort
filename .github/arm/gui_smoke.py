"""Opens the Kilosort GUI for a few seconds and closes it."""
import sys

import pyqtgraph as pg
from qtpy import QtCore, QtWidgets

from kilosort.gui import DarkPalette, KilosortGUI

app = QtWidgets.QApplication(sys.argv)
app.setStyle('Fusion')
app.setPalette(DarkPalette())
pg.setConfigOption('useOpenGL', True)
gui = KilosortGUI(app, skip_load=True)
gui.show()
QtCore.QTimer.singleShot(5000, app.quit)
app.exec_()
print('GUI opened and closed OK, Qt platform:', app.platformName())
