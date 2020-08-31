for /d %%d in (testrun_*) do rmdir /s /q %%d
del test_*.exe
del output.exe output.pdb
