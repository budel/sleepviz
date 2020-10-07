import csv

def mergeCSV(basefile, newfile):
    sleep1 = loadCSV(basefile)
    sleep2 = loadCSV(newfile)

    for s2 in sleep2:
        if s2['start'] in [s1['start'] for s1 in sleep1] and s2['stop'] in [s1['stop'] for s1 in sleep1]:
            continue
        #TODO add overlapping test
        sleep1.append(s2)
    
    sleep1 = sorted(sleep1, key=lambda x: x['start'])
    sleep1[1:], sleep1[0] = sleep1[:-1], sleep1[-1]  # put header back to front
    
    with open(basefile, 'w') as csvfile:
        mywriter = csv.writer(csvfile)
        for s1 in sleep1:
            mywriter.writerow([s1['sid'], s1['start'], s1['stop'], s1['rating']])
            

def loadCSV(file):
    sleep = []
    
    with open(file, 'r') as csvfile:
        myreader = csv.reader(csvfile)
        row = next(myreader)  # gets the first line
        if row[0] != 'sid' or row[1] != 'start' or row[2] != 'stop' or row[3] != 'rating':
             raise Exception("Wrong csv header")

    with open(file, 'r') as csvfile:
        myreader = csv.reader(csvfile)
        for i, row in enumerate(myreader):
            sleep.append({})
            sleep[i]['sid'] = row[0]
            sleep[i]['start'] = row[1]
            sleep[i]['stop'] = row[2]
            sleep[i]['rating'] = row[3]
    return sleep
