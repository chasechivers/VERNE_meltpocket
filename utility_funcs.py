import dill as pickle
import re
import os
import numpy as np


def nat_sort(x):
	convert = lambda text: int(text) if text.isdigit() else text.lower()
	alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
	return sorted(x, key=alphanum_key)


def file_namer(model, outputdir, *argv):
	# defaults chosen for file so that the file name isn't enormous...
	#  variables chosen to be model.__ can be changed to be included in file name if so desired
	# wants or needs...
	wants = {"D": 50, "w": model.vehicle_r * 2.5, "dx": 0.01, "dz": 0.1, "dt": 10, "kT": True, "Z0": None,
	         "vehicle_r":0.3, "vehicle_h":4, "vehicle_T": 300, "melt_mod": 1, "Tsurf": 110,
	         "composition": None, "concentration": None}

	dict = {key: value for key, value in model.__dict__.items() if key in wants and value != wants[key]}

	#dict = {key: value for key, value in model.__dict__.items() if not key.startswith('__') and \
	#        not callable(key) and type(value) in [str, bool, int, float] and value != defaults[key]}
	file_name = outputdir
	if model.verbose: print('naming file!')

	def string_IO(input):
		if isinstance(input, bool):
			if input is True:
				return 'on'
			else:
				return 'off'
		else:
			return input

	if len(argv) != 0:
		print('custom naming!')
		for var in argv:
			print(f" adding {var} to filename")
			if var in dict.keys():
				file_name += f"_{var}={dict[var]}"
			elif var in model.__dict__.keys():
				if isinstance(var, (float, int)):
					file_name += f"_{var}={model.__dict__[var]:0.03f}"
				else:
					file_name += f"_{var}={model.__dict__[var]}"
			else:
				file_name += f"_{var}"
	else:
		for key in dict.keys():
			if isinstance(dict[key], float):
				if "vehicle" in key: key.replace("vehicle", "ve")
				file_name += f"_{key}={dict[key]:0.03f}"
			else:
				if "vehicle" in key: key.replace("vehicle", "ve")
				file_name += f"_{key}={key, string_IO(dict[key])}"
	return file_name + ".pkl"


def save_data(data, file_name, outputdir, final=1, *argv):
	if isinstance(data, (dict, list)) or type(data).__module__ == np.__name__:
		file_name = outputdir + file_name + '.pkl'
	elif final == 1:
		file_name = file_namer(data, outputdir + file_name, *argv)
	elif final == 0:
		file_name = outputdir + file_name
	with open(file_name, 'wb') as output:
		pickle.dump(data, output, -1)
		output.close()


def load_data(file_name):
	if ".pkl" in file_name:
		with open(file_name, 'rb') as input:
			return pickle.load(input)
	else:
		raise Exception("File extension not recognized")


def directory_spider(input_dir, path_pattern="", file_pattern="", maxResults=500000):
	'''
	Returns list of paths to files given an input_dir, path_pattern, and file_pattern
	'''
	file_paths = []
	if not os.path.exists(input_dir):
		raise FileNotFoundError("Could not find path: %s" % (input_dir))
	for dirpath, dirnames, filenames in os.walk(input_dir):
		if re.search(path_pattern, dirpath):
			file_list = [item for item in filenames if re.search(file_pattern, item)]
			file_path_list = [os.path.join(dirpath, item) for item in file_list]
			file_paths += file_path_list
			if len(file_paths) > maxResults:
				break
	return file_paths[0:maxResults]


def untar_file(tarfilename, outdir, which_file='_'):
	import tarfile
	try:
		print('Opening file:', tarfilename)
		t = tarfile.open(tarfilename, 'r')
		print('... File opened')
	except IOError as e:
		print(e)
	else:
		if which_file == '_':
			print('Extracting all files to', outdir)
			t.extractall(outdir)
			print('Finished extracting')
			filelist = [member.name for member in t.getmembers()]
		else:
			if 'md' in which_file:
				print('Matching model file to results file in zip')
				idx = which_file.find('runID')
				NID = which_file[idx:idx+9]
				for dirpath, dirname, filename in os.walk(outdir+'/results/'):
					filelist = [item for item in filename if NID in item]
					if len(filelist) > 0:
						print('File already extracted:\n\t', filelist)
						return filelist[0]
				extract_this = [member for member in t.getmembers() if NID in member.name]
				print('Extracting', extract_this, 'to', outdir)
				t.extractall(outdir, members=extract_this)
				if type(extract_this) is list:
					filelist = extract_this[0].name
				elif type(extract_this) is tarfile.TarInfo:
					filelist = extract_this.name
			else:
				print('Extracting files with', which_file, 'to', outdir)
				extract_this = [member for member in t.getmembers() if which_file in member.name]
				print(' ... Extracting', extract_this)
				t.extractall(outdir, members=extract_this)
				if type(extract_this) is list:
					filelist = extract_this[0].name
				elif type(extract_this) is tarfile.TarInfo:
					filelist = extract_this.name
	t.close()
	return filelist
