#!C:\Users\Manish\PycharmProjects\SC\venv\Scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'imageio==2.4.1','console_scripts','imageio_remove_bin'
__requires__ = 'imageio==2.4.1'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('imageio==2.4.1', 'console_scripts', 'imageio_remove_bin')()
    )
