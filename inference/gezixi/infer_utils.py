from StarCC import PresetConversion
import requests
import re

cn2hk_converter = PresetConversion(src='cn', dst='hk', with_phrase=False)

def create_cn2tl_dict():
	with open('inference/gezixi/new_zh_minnan_dict.csv', 'r', encoding='utf-8') as f:
		items = f.readlines()

	c2t_dict = {}
	for item in items:
		try:
			cn, tl = item.strip().split(',')
		except:
			#print(item)
			continue
		tl = re.sub('-+', ' ', tl).strip()
		if cn not in c2t_dict:
			c2t_dict[cn] = tl 
	return c2t_dict

def create_tl2ipa_dict():
	with open('inference/gezixi/new_TL2IPA.txt', 'r', encoding='utf-8') as f:
		items = f.readlines()
	tl2ipa = {}
	for item in items:
		tl, py = item.strip().split('\t')
		tl2ipa[tl] = '|'.join(py.split(' '))
	return tl2ipa

def create_subyunmu_dict():
	with open('inference/gezixi/sub_yunmu.txt', 'r', encoding='utf-8') as f:
		items = f.readlines()
	subyunmu = {}
	for item in items:
		yunmu, _, sub = item.strip().split(' ')
		subyunmu[yunmu] = sub
	return subyunmu

def create_ipa2tl_dict():
    with open('inference/gezixi/ipa2tl.txt', 'r', encoding='utf8') as f:
        lines = f.readlines()
    ipa2tl = {}
    for line in lines:
        k,v = line.strip().split(' ')
        ipa2tl[v] = k
    return ipa2tl

def default_trans(cn, mode='mm'):
	c2t_dict = create_cn2tl_dict()
	if mode == 'mm':
		return mm_trans(c2t_dict, cn)
	else:
		return vanilla_trans(c2t_dict, cn)

def vanilla_trans(c2t_dict, sen):
	'''
	translate word by word
	:param: sen: simplified Chinese sentence
	'''
	cn_list = [i for i in sen]
	tl = []
	for t_c in cn_list:
		tl.append(c2t_dict.get(t_c, 'UNK'))
	return ' '.join(tl)

def mm_trans(c2t_dict, sen):
	'''
	translate according to the principle of maximum matching
	:param: sen: simplified Chinese sentence
	'''
	tl_list = []
	offset = 0
	end = len(sen)
	while offset < end:
		for e in range(end, offset, -1):
			if sen[offset:e] in c2t_dict:
				tl_list.append(c2t_dict[sen[offset:e]])
				offset = e
				break
			elif e - offset == 1 and sen[offset:e] not in c2t_dict:
				print(f'{sen[offset:e]} not in dict')
				tl_list.append('UNK')
				offset += 1
	tl_sen = ' '.join(tl_list)

	return tl_sen

def tl_to_ipa(tl):
	'''
	convert tl to ipa phonemes
	:parmas: tl: tl sentence
	'''
	tl2ipa_dict = create_tl2ipa_dict()
	tl_list = tl.split(' ')
	ipa_sen = []
	for tl_token in tl_list:
		if tl_token.startswith('j'):
			tl_token = tl_token.replace('j', 'l')
		ipa_sen.append(tl2ipa_dict.get(tl_token, 'UNK'))
	return ' '.join(ipa_sen)

def ipa_to_tlpy(ipa):
	'''
	convert ipa phonemes to tl pinyin
	:parmas: tl: ipa phoneme sequence
	'''
	ipa2tl_dict = create_ipa2tl_dict()
	ipa_list = ipa.split(' ')
	tlpy_sen = []
	for ipa_token in ipa_list:
		pys = ipa_token.split('|')
		tlpy = []
		for py in pys:
			tlpy.append(ipa2tl_dict.get(py, 'UNK'))
		tlpy_sen.append('|'.join(tlpy))
	return ' '.join(tlpy_sen)


def cn_to_tl(cn):
    hk = cn2hk_converter(cn)
    url='http://tts001.iptcloud.net:8804/display'
    kw = {
        'text0':hk
    }
    header = {
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.51'
    }
    try:
        tl = requests.get(url, kw, headers=header, timeout=2).text
        tl = re.sub(r'[0-9]+', '', tl)
        tl = re.sub(r'-+', ' ', tl)
        tl = re.sub(r'\.+', '', tl)
        hk_lst = [i for i in hk]
        tl_lst = tl.split(' ')
        if len(hk_lst) != len(tl_lst):
            print(f'Wrong translation: \n cn: {cn} \n tl: {tl} \n use maxium matching translation')
            tl = default_trans(cn, mode='mm')
            tl_lst = tl.split(' ')
            if len(hk_lst) != len(tl_lst):
                print(f'Wrong translation: \n cn: {cn} \n tl: {tl} \n use vanilla translation')
                tl = default_trans(cn, mode='vanilla')
                tl_lst = tl.split(' ')
                assert len(hk_lst) == len(tl_lst)
            print(f"Translated TL: {tl}")
    except:
        print("Net err")
        tl = default_trans(cn, mode='mm')
        hk_lst = [i for i in hk]
        tl_lst = tl.split(' ')
        if len(hk_lst) != len(tl_lst):
            print(f'Wrong translation: \n cn {cn} \n tl {tl} \n use vanilla translation')
            tl = default_trans(cn, mode='vanilla')
            tl_lst = tl.split(' ')
            assert len(hk_lst) == len(tl_lst)
        print(f"Translated TL: {tl}")

    ipa = tl_to_ipa(tl)
    tl_py = ipa_to_tlpy(ipa)
    return tl_py

def curate_text(text_raw_list):
    text_cn = []
    text_len = []
    text_pos = []
    mark = -1
    for idx, token in enumerate(text_raw_list):
        if token == 'sp':
            continue
        elif token == '-':
            text_len[mark] += 1
        else:
            text_cn.append(token)
            text_len.append(1)
            text_pos.append(idx)
            mark += 1
    assert len(text_cn) == len(text_len)
    tl_py = cn_to_tl(''.join(text_cn))
    subyunmu_dict = create_subyunmu_dict()
    ext_tl_py_list = text_raw_list[:]
    for idx, pinyin in enumerate(tl_py.split(' ')):
        t_len = text_len[idx]
        t_pos = text_pos[idx]
        if ext_tl_py_list[t_pos] == 'sp':
            continue 
        else:
            yunmu = pinyin.split('|')[-1]
            ext_tl_py_list[t_pos] = pinyin
            if t_len > 1:
                ext_tl_py_list[t_pos + 1:t_pos + t_len] = [subyunmu_dict[yunmu]] * (t_len -1)
    return ext_tl_py_list

if __name__ == '__main__':
	cn = 'Sp 此 去 - - 云 南 sp 路 - 千 - -  - 里 - sp'
	text_raw = cn.strip().lower().split(' ')
	text_raw_list = [i for i in text_raw if i != '']
	print(text_raw_list)
	ext_tl_py_list = curate_text(text_raw_list)
	print(ext_tl_py_list)