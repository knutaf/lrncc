for /d %%d in (testrun_*) do rmdir /s /q %%d
del test_*.exe test_*.pdb output.exe output.pdb output.ilk *.lnk *.obj
