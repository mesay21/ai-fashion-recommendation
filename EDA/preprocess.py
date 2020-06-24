import json
import pandas as pd
import os


#[x[0] for x in os.walk(directory)]

with open('IQON3000/2649573/4070026/4070026.json', 'r') as myfile:
    data=myfile.read()
obj = json.loads(data)

print(obj.keys())
for item in obj['items']:
    print(item.keys())
    print("CATEGORYS #############", item['categorys'])
    print("expressions #############", item['expressions'])
    print("category x color ##########", item['category x color'])
#print(obj['items'][1])


from json.decoder import JSONDecodeError

IQON_SET_IDS = []
final_results = []
count = 0
for x in os.walk('IQON3000'):
    
    for pos_json in os.listdir(x[0]):
        
        if pos_json.endswith('.json'):
            count = count + 1
            #print(x[0] + '/' + pos_json)
            #with open(x[0] + '/' + pos_json, 'r') as myfile:
            #    data=myfile.read()
            #obj = json.loads(data)
            
            #with open(x[0] + '/' + pos_json, encoding='utf-8', errors='ignore') as json_data:
            #    obj = json.load(json_data, strict=False)
                


            with open(x[0] + '/' + pos_json, 'r') as infile:
                try:
                    obj = json.load(infile)
                    #data = old_data + obj
                    #json.dump(data, outfile)
                except JSONDecodeError:
                    pass
            #print(data)
            
            item_lst = list(obj.keys())
            
            a_set = {}
            
            a_set["set_id"] = obj['setId']
            
            IQON_SET_IDS.append(obj['setId'])
            
            a_set["items"] = []
            a_set["user_id"] = obj['user']
            a_set["dataset"] = "iqon"
            for i in range(len(obj['items'])):
                temp_dic = {}
                temp_dic['index'] = obj['items'][i]['itemId']
                temp_dic['name'] = 'casual'
                a_set["items"].append(temp_dic)
    
            final_results.append(a_set)
        
        if count == 20000:
            print(final_results)
            with open("iqon.json", "w") as outfile: 
                json.dump(final_results,outfile)
            break
    if count == 20002:
        break
        
    
#print(final_results)

                

            

            
    #print(x[0].split('/'))
    #print(instance)
with open('train_no_dup.json', 'r') as infile:
    polyvore = json.load(infile)

print(len(polyvore))

POLYVORE_SET_IDS = []
for sets in polyvore:
    POLYVORE_SET_IDS.append(sets['set_id'])


with open('iqon.json', 'r') as infile:
    iqon = json.load(infile)

print(len(iqon))

result = iqon + polyvore


with open('train.json', 'w') as out:
    json.dump(result,out)

for i in POLYVORE_SET_IDS:
    if i in IQON_SET_IDS:
        print("TRUE")
