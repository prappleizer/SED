# SED
Repository of code used to calculate composite SEDs and run prospector fits using FSPS.


Prescription for usage (for Step One- construction of full composite SEDs)

1) run the 'new_sed_version.py' file in the /code directory and specify a new version number (can by anything, but should presumably be a number greater than the number of versions in the /SEDs directory. e.g., 'python new_sed_version.py 3' in terminal would produce a /v3 directory and all needed subdirectories within the SEDs directory.) 

2) With the empty directories initialized, open main_2.py and after any adjustments are made to how the SEDs are created (the only reason to make a new version), change the output_version_number variable in the filepaths to the new version number within the string. Also open filter_split.py and change the 'composite_filter_output_directory' variable (at top) to point to the proper version directory.

3) In python, run filter_split.py. This should fill the currently empty composite_filters/iteration___/ directories with the previous optical sed filters.

4) In python, run 'main_v2.py' and then the function run_main(num_stacks) where num_stacks is the number of FIR stacks you want to split the Herschel data into. This will populate the /SEDs/v__/seds directory with 32 new sed files (with columns for wavelength, flux, and error (flux and error in maggies)). It will also create composite filters for the stacks it generates for MIPS and the FIR points (number based on how many stacks you input), and adds them into the appropriate composite_filters/iteration__/ directories (the filter numbers increase monotonically from 1000, the new IR stack filters are the last several once you do this.)

5) You now have the sed files and filter files needed to run a prospector fit on any of the 32 SEDs.