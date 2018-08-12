from bs4 import BeautifulSoup
import requests
import re
import sys
import os

inputs ={}
outputs = {}
ios = {}
docs = {}

def get():
	if os.path.isfile("cache/docs.html"):
		with open("cache/docs.html", 'r') as f:
			print("Using cache", file=sys.stderr)
			return f.read()
	r = requests.get("http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html")
	with open("cache/docs.html", 'w') as f:
		f.write(r.text)
	return r.text

def main():
	txt = get()
	soup = BeautifulSoup(txt, "html5lib")
	contents = soup.find_all(id="api-introduction")
	topics = contents[0].find_all(class_="topic concept nested1")
	for topic in topics:
		rawFnName = topic.find_all(class_='title topictitle2')[0].text
		try:
			fnName = re.search('cudnn.+$', rawFnName).group(0)
		except AttributeError as e:
			print("rawFnName: {}".format(rawFnName), file=sys.stderr)
			continue
		try:
			paramsDL = topic.find_all(class_='dl')[0]      # first definition list is params
		except IndexError:
			print("rawFnName: {} - topic has no dl class".format(fnName), file=sys.stderr)
			continue

		# check previous
		if paramsDL.previous_sibling.previous_sibling.text != "Parameters":
			print("rawFnName: {} has no params::: {}".format(fnName, paramsDL.previous_sibling), file=sys.stderr)
			continue

		params = paramsDL.find_all(class_='dt dlterm') # name 
		paramsDesc = paramsDL.find_all(class_='dd')    # use type
		paramUse = []
		for d in paramsDesc:
			try:
				use = d.find_all(class_='ph i')[0].text
			except IndexError as e:
				use = "Input"
			paramUse.append(use)


		if len(params) != len(paramUse):
			print("rawFnName: {} - differing params and use cases".format(fnName), file=sys.stderr)
			continue

		inputParams = [p.text.strip() for i, p in enumerate(params) if (paramUse[i].strip()=='Input') or (paramUse[i].strip()=="Inputs")]
		outputParams = [p.text.strip() for i, p in enumerate(params) if (paramUse[i].strip()=='Output') or (paramUse[i].strip()=="Outputs")]
		ioParams = [p.text.strip() for i, p in enumerate(params) if paramUse[i].strip()=='Input/Output']

		inputs[fnName] = inputParams
		outputs[fnName] = outputParams
		ios[fnName] = ioParams

		# extract docs
		try:
			docbody = topic.find_all(class_='body conbody')[0]
		except IndexError:
			print("fnName: {} - no body".format(fnName), file=sys.stderr)
			continue
		# clear is better than clever. 
		doc = docbody.find_all("p")[0].text
		doc = doc.replace("\n", "")
		doc = re.sub("\t+", " ", doc)
		doc = re.sub("\s+", " ", doc)
		doc = doc.replace('"', '`')
		doc = doc.replace("This function", fnName)
		doc = doc.replace("This routine", fnName)
		doc = doc.replace("This", fnName)
		doc = doc.strip()
		docs[fnName] = doc

	# write the go file
	print("package main")
	print("var inputParams = map[string][]string{")
	for k, v in inputs.items():
		if len(v) == 0: continue
		print('"{}": {{ '.format(k), end="")
		for inp in v :
			split = inp.split(",")
			for s in split:
				print('"{}", '.format(s.strip()), end="")
		print("},")
	print("}")

	print("var outputParams = map[string][]string{")
	for k, v in outputs.items():
		if len(v) == 0: continue
		print('"{}": {{ '.format(k), end="")
		for inp in v :
			split = inp.split(",")
			for s in split:
				print('"{}", '.format(s.strip()), end="")
		print("},")
	print("}")

	print("var ioParams = map[string][]string{")
	for k, v in ios.items():
		if len(v) == 0: continue
		print('"{}": {{ '.format(k), end="")
		for inp in v :
			split = inp.split(",")
			for s in split:
				print('"{}", '.format(s.strip()), end="")
		print("},")
	print("}")

	print("var docs = map[string]string{")
	for k, v in docs.items():
		print('"{}": "{}",'.format(k, v.strip()))
	print("}")

main()