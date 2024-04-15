from django.shortcuts import render
from django.http import HttpResponse

from django.shortcuts import render
import numpy as np 
import pickle 
from sklearn.pipeline import Pipeline
from StartupAacquisition.settings import BASE_DIR
from os.path import join 
import numpy as np 
PATH = join(BASE_DIR  , 'PipelineModel/model.pkl') ; 


def home(request):
    pred = None
    if(request.method=='POST'):
        model = pickle.load(open(PATH ,'rb')) ; 
        data = np.array([request.POST['Founded'] , request.POST['funding_first'] ,request.POST['funding_last'] , request.POST['funding_rounds'] ,request.POST['FundingUSD'] ,request.POST['MilestoneFirst'] , request.POST['MilestoneLast'] , request.POST['no_milestone'] , request.POST['relationships']] , dtype=np.int16)
        data = data.reshape(1 , 9) 
        pred = model.predict(data) 
        if pred==0:
            pred  = 'Operating'
        elif pred==1 :
            pred = 'Accquired'
        elif pred ==2 :
            pred = 'Closed' 
        else:
            pred = 'Ipo'
        # print(pred.shape) 
        # print(request.POST) ; 
    content = {
        'output' : pred
    }
    
    return render(request , 'index.html', context = content )