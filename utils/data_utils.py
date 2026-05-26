import pyarrow.parquet as pq
import os
import re
import requests


def read_parquet(parquet_path):
	table = pq.read_table(parquet_path)
	return [dict(zip(table.schema.names, row)) for row in zip(*[table[column].to_pylist() for column in table.schema.names])]


def get_mm_cols(concept, klist):
    conc_dict = concept._asdict()
    conc_list = [conc_dict.get(kk) for kk in klist]
    return(tuple(conc_list))


def get_mm_dict(concept, klist):
    conc_dict = concept._asdict()
    return {kk: conc_dict.get(kk) for kk in klist}


def get_umls_cui_page(cui, subpage='', version='current'):
	base_url = f"https://uts-ws.nlm.nih.gov/rest/content/{version}/CUI/{cui}"
	if subpage:
		base_url += f'/{subpage}'
	return base_url


def umls_url_request(api_key, url):
	query = {'apiKey': api_key}
	try:
		response = requests.get(url, params=query)
		response.raise_for_status()
	except requests.exceptions.RequestException as e:
		print(f"ERROR: {e}")
		return None
	return response.json()


def get_info_cui(msg):
	url = msg.get('result', {}).get('concept', '')
	return get_url_cui(url)


def get_url_cui(url):
	pattern = r'/CUI/([A-Z0-9]+)'
	match = re.search(pattern, url)
	if match:
		return match.group(1)
	return None


def url_request_cui(url):
	info = umls_url_request(url)
	if info:
		return get_info_cui(info)
	return None


def cui_request_name(api_key, cui, version='2020AA'):
	umls_url = get_umls_cui_page(cui, version=version)
	info = umls_url_request(api_key, umls_url)
	if info:
		name = info.get('result', {}).get('name', '')
		return name
	return ''


def entity_weight_list_to_string(entity_weight_list):
    value = "\n<<ENTITY INFO>>"
    for ew in entity_weight_list:
        value += f"{ew[0]} @@@ {ew[1]}\n"
    value += "<<ENTITY INFO END>>\n"
    return value


def result_string_parse(result_str, tag='ENTITY INFO'):
	start_tag, end_tag = f'<<{tag}>>', f'<<{tag} END>>' 
	pattern = rf"{start_tag}(.*?){end_tag}"
	match = re.search(pattern, result_str, re.DOTALL | re.MULTILINE)
	if match:
		if tag == 'ENTITY INFO':
			entity_weight_list = []
			entity_info = match.group(1).strip()
			lines = entity_info.split('\n')
			for line in lines:
				parts = line.split('@@@')
				if len(parts) == 2:
					entity = parts[0].strip()
					try:
						weight = float(parts[1].strip())
					except ValueError:
						weight = 0.0
					entity_weight_list.append([entity, weight])
			return entity_weight_list
		elif tag == 'RESULT':
			return match.group(1).strip('\n').strip()
	return None


def cui_request_relation_triples(api_key, cui, version='2020AA'):
	umls_url = get_umls_cui_page(cui, subpage='relations', version=version)
	cui_name_dict = { cui: cui_request_name(api_key, cui, version=version) }
	relation_info = umls_url_request(api_key, umls_url)
	triple_list = []
	for item in relation_info.get('result', []):
		url2 = item.get('relatedId', '')
		if url2:
			# 是否直接连接到概念CUI，若是，直接提取
			cui2 = get_url_cui(url2)
			if not cui2:
				# 若连接到的是原子概念，则需要再请求一�?				relation_info2 = umls_url_request(api_key, url2)
				cui2 = get_info_cui(relation_info2) if relation_info2 else None
			if cui == cui2 or not cui2:
				continue
			cui2_name = cui_name_dict.get(cui2, None)
			if not cui2_name:
				cui_name_dict[cui2] = cui_request_name(api_key, cui2, version=version)
				cui2_name = cui_name_dict[cui2]
			# print(f'Processing relation URL: {url2}')
			# print(f'Found related CUI: {cui2}, Name: {cui2_name}')
			# exit()
			relation_label = item.get('relationLabel', '')
			relation_addtional_label = item.get('additionalRelationLabel', '')
			# print(f'Relation: {relation_label}, Additional Relation: {relation_addtional_label}')
			triple_list.append({
				'subject_cui': cui,
				'subject_name': cui_name_dict.get(cui, ''),
				'object_cui': cui2, 
				'object_name': cui2_name,
				'relation': relation_label, 
				'additional_relation': relation_addtional_label
			})
	# 去除完全相同的triple
	triple_list = [dict(t) for t in {tuple(d.items()) for d in triple_list}]
	return triple_list
