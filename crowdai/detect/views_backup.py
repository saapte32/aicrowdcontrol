from django.shortcuts import render
from .models import tbl_Incident_Master,tbl_incident_Trans_Mask,tbl_incident_Trans_SD

from django.http import HttpResponse, JsonResponse

from django.http import Http404
import json
def index(request):

    return render(request,'detect/dashboard.html')

# Create your views here.


def violations(request):
    try:
       # violations = tbl_Incident_Master.objects.get(pk=question_id)

       violations = tbl_Incident_Master.objects.all()
       totalviolations = []
       #import pdb;pdb.set_trace()
       print(type(violations))
       for violation in violations.iterator():
            selectedviolations = {}

            selectedviolations['index'] =violation.incidentId
            selectedviolations['maskcount'] = 0
            selectedviolations['sdcount'] = 0
            print(violation.__dict__)
            print(violation.incidentId)
            selectedviolations['date']=violation.Date_Time_frm.date()
            #sdincidents = tbl_incident_Trans_SD.objects.all()
            if violation.IncidentType=='SD':
                sdincidents = tbl_incident_Trans_SD.objects.filter(incidentId=violation.incidentId)
                # selectedviolations['index']+=1
                for sdincident in sdincidents.iterator():
                    print(sdincident.ObjectId)
                    selectedviolations['sdcount']=sdincident.ObjectId


            if violation.IncidentType=='FM':
                maskincidents = tbl_incident_Trans_Mask.objects.filter(incidentId=violation.incidentId)
                # selectedviolations['index'] += 1
                for maskindicent in maskincidents.iterator():
                    print(maskindicent.__dict__)
                    selectedviolations['maskcount']=maskindicent.ObjectId

            if not selectedviolations['maskcount']:
                selectedviolations['maskcount']=0
            if not selectedviolations['sdcount']:
                selectedviolations['sdcount']=0
            selectedviolations['totalcount']=int(selectedviolations['sdcount'])+int(selectedviolations['maskcount'])
            print("my selected violations")
            print(selectedviolations)
            totalviolations.append(selectedviolations)


           #maskincident=tbl_incident_Trans_Mask.objects.filter(IncidentId_id=violation.incidentId)
       # for violation in violations:
       #     sdincident=tbl_incident_Trans_SD.objects.filter(IncidentId_id=violation.incidentId)
       #     print(sdincident.ObjectId)
       print("Selected violations")
       print(totalviolations)
       return render(request, 'detect/dashboard.html', {"violations": totalviolations})


    except violations.DoesNotExist:
        raise Http404("Violation does not exist")

    #return render(request,'distancedetector/dashboard.html',{"violations":violations})



    #totalviolations.append(selectedviolations)




def sdclick(request,sdval=1):
    print("sd clicked",sdval)
    #payload = {'data':'/static/assets/indmarket.mp4'}
    #payload = {'data': 'assets/indmarket.mp4'}
    data="/static/assets/indmarket.mp4"
    path_details = {}
    path_details={"vidurl":"/static/assets/indmarket.mp4"}
    #return HttpResponse(json.dumps(payload), content_type='application/json')
    #return JsonResponse({'success':True,'data':'/static/assets/indmarket.mp4'})
    #return JsonResponse(json.loads(json.dumps({'data':'/static/assets/indmarket.mp4'})))
    #return render(request, 'detect/dashboard.html', {"payload": payload})
    #return render(request, 'detect/dashboard.html', {"data": data})
    #return HttpResponse("/static/assets/indmarket.mp4")
    return JsonResponse({'path_details': path_details})
    #return JsonResponse({'success': True, 'data': '/static/assets/indmarket.mp4'})



def maskclick(request,maskval=1):
    print("sd clicked", maskval)
    path_details = {"vidurl": "/static/assets/indmarket.mp4"}
    return JsonResponse({'path_details': path_details})



def clicksd(request,fordate='1',cameraid=1):
    print("sd clicked 2 para",fordate,cameraid)
    path_details = {"vidurl": "/static/assets/output.mp4"}
    return JsonResponse({'path_details': path_details})



def clickmask(request,fordate='1',cameraid=1):
    print("mask clicked 2 para",fordate,cameraid)
    path_details = {"vidurl": "/static/assets/video_name_FM1.mp4"}
    return JsonResponse({'path_details': path_details})