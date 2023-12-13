PHN_MAP = {
	'a': ['a', 'aʔ', 'ak', 'ap', 'at', 'aⁿ', 'aⁿʔ'], 
	'ai': ['ai', 'aiʔ', 'aⁿiⁿ', 'aⁿiⁿʔ'], 
	'am': ['am'],
	'an': ['an'],
	'aŋ': ['aŋ'],
	'au': ['au', 'auʔ', 'aⁿuⁿ', 'aⁿuⁿʔ'],
	'e': ['e', 'eʔ', 'eⁿ', 'eⁿʔ'],
	'i': ['i', 'iʔ', 'ik', 'iⁿ', 'iⁿʔ', 'ip', 'it'],
	'ia': ['ia', 'iaʔ', 'iak', 'iⁿaⁿ', 'iⁿaⁿʔ', 'iap', 'iet'],
	'iam': ['iam'],
	'ien': ['ien'],
	'iaŋ': ['iaŋ'],
	'iau': ['iau', 'iauʔ', 'iⁿaⁿuⁿ', 'iⁿaⁿuⁿʔ'],
	'im': ['im'],
	'iŋ': ['in', 'iŋ'],
	'iə': ['iə', 'iəʔ'],
	'iok': ['iok', 'io', 'ioʔ', 'iⁿoⁿ'],
	'ioŋ': ['ioŋ'],
	'iu': ['iu', 'iuʔ', 'iⁿuⁿ', 'iⁿuⁿʔ', 'iut'],
	'ə': ['ə', 'əʔ'],
	'o': ['o', 'ok', 'oʔ', 'oⁿ', 'oⁿʔ', 'op'],
	'om': ['om'],
	'oŋ': ['oŋ'],
	'u': ['u', 'uʔ', 'ut'],
	'ua': ['ua', 'uaʔ', 'uak', 'uⁿaⁿ', 'uⁿaⁿʔ', 'uat'],
	'uai': ['uai', 'uaiʔ', 'uⁿaⁿiⁿ', 'uⁿaⁿiⁿʔ'],
	'uaŋ': ['uan', 'uaŋ'],
	'ue': ['ue', 'ueʔ', 'uⁿeⁿ'],
	'ui': ['ui', 'uiʔ', 'uⁿiⁿ', 'uⁿiⁿʔ'],
	'un': ['un'],
	'l': ['j'],
	'p': ['b', 'p'],
	'k': ['g', 'k']
}

def make_phn_map_dict(yunmu_map):
	PHN_REDUCE_DICT = {}
	for k, v in yunmu_map.items():
		for ym in v:
			PHN_REDUCE_DICT[ym] = k
	return PHN_REDUCE_DICT
	
if __name__ == '__main__':
	PHN_REDUCE_DICT = make_phn_map_dict(PHN_MAP)
	print(PHN_REDUCE_DICT)
