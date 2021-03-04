from django.shortcuts import render
from .models import tbl_Incident_Master
from django.db.models import Sum
from django.http import HttpResponse, JsonResponse
from django.http import Http404
from datetime import date, timedelta
from django.db.models import F, Sum, IntegerField
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from .models import LocationMaster
from .models import CameraMaster

# from .live import feedlive

def index(request):
    return render(request, 'detect/dashboard.html')


# Create your views here.


def violations(request):
    # violations = tbl_Incident_Master.objects.get(pk=question_id)
    request.session['cameraID']=1
    # cameraid=1
    # if(request.session.get('cameraID')):
    #     cameraid=request.session.get('cameraID')
    # else:
    #     request.session['cameraID']=1
    #     cameraid= request.session['cameraID']
    BaseQueryfilter = tbl_Incident_Master.objects.filter(CameraId=request.session.get('cameraID'))
    last7DaysSD = []
    last7DaysFM = []
    for i in range(1, 8):
        days_before = (date.today() - timedelta(days=i)).isoformat()
        lastSD = BaseQueryfilter.filter(Date_Time_frm__date=days_before).aggregate(Sum('SocialDistance'))
        if lastSD['SocialDistance__sum'] == None:
            lastSD['SocialDistance__sum'] = 0
        last7DaysSD.append(lastSD['SocialDistance__sum'])

        lastFM = BaseQueryfilter.filter(Date_Time_frm__date=days_before).aggregate(Sum('FaceMask'))
        if lastFM['FaceMask__sum'] == None:
            lastFM['FaceMask__sum'] = 0
        last7DaysFM.append(lastFM['FaceMask__sum'])



    # violations = tbl_Incident_Master.objects.all()



    last7DaysSD.reverse()
    last7DaysFM.reverse()
    # import pdb;pdb.set_trace()

    # selectedviolations['totalcount'] = int(selectedviolations['sdcount']) + int(selectedviolations['maskcount'])
    # for violation in violations.iterator():
    #      selectedviolations = {}
    #      #selectedviolations['index'] = 0
    #      selectedviolations['id']=violation.id
    #      selectedviolations['date']=violation.Date_Time_frm
    #      #sdincidents = tbl_incident_Trans_SD.objects.all()
    #      #sdincidents = tbl_Incident_Master.objects.filter(id=violation.id)
    #      #selectedviolations['sdcount'] = tbl_Incident_Master.objects.filter()
    #      #selectedviolations['index']+=1
    #      for sdincident in sdincidents.iterator():
    #          selectedviolations['sdcount']=sdincident.SocialDistance
    #          selectedviolations['maskcount'] = sdincident.FaceMask
    #      selectedviolations['totalcount']=int(selectedviolations['sdcount'])+int(selectedviolations['maskcount'])
    #
    #      totalviolations.append(selectedviolations)





    # maskincident=tbl_incident_Trans_Mask.objects.filter(IncidentId_id=violation.incidentId)
    # for violation in violations:
    #     sdincident=tbl_incident_Trans_SD.objects.filter(IncidentId_id=violation.incidentId)
    #     print(sdincident.ObjectId)
    violationrecords = violationcount(request, BaseQueryfilter)

    # import pdb;pdb.set_trace()
    return render(request, 'detect/dashboard.html', {"violations": violationrecords,
                                                     "last7DaysSD": last7DaysSD,
                                                     "last7DaysFM": last7DaysFM
                                                     })




    # return render(request,'distancedetector/dashboard.html',{"violations":violations})



    # totalviolations.append(selectedviolations)

def violationcount(request,BaseQueryfilter):

    day = (date.today() - timedelta(days=0)).isoformat()
    SD = BaseQueryfilter.filter(Date_Time_frm__date=day).annotate(
        totalViolations=Sum(F('SocialDistance') + F('FaceMask'), output_field=IntegerField()))
    #print("lenSD" ,len(SD))
    #if(len(SD)>0):
    page_no = request.GET.get('page', 1)
    records_per_page = 10
    paginator = Paginator(SD, records_per_page)

    try:
        violationrecords = paginator.page(page_no)
    except PageNotAnInteger:
        violationrecords = paginator.page(1)
    except EmptyPage:
        violationrecords = paginator.page(paginator.num_pages)

    return violationrecords

def getviolationscount(request):
    #print("am here")
    if (request.session.get('cameraID')):
        cameraid = request.session.get('cameraID')
    else:
        request.session['cameraID'] = 1
        cameraid =request.session.get('cameraID')

    BaseQueryfilter = tbl_Incident_Master.objects.filter(CameraId=cameraid)
    violationrecords = violationcount(request,BaseQueryfilter)
    #import pdb;pdb.set_trace()

    # else:
    #     violationrecords={}
    #     print("all zeros")
    #     violationrecords['Date_Time_frm']=0
    #     violationrecords['FaceMask'] = 0
    #     violationrecords['SocialDistance'] = 0
    #     violationrecords['totalViolations'] = 0
    #     return violationrecords

    return render(request, 'detect/violations.html', {"violations": violationrecords

                                                      })


def sdclick(request, sdval=1):
    # print("sd clicked",sdval)
    # payload = {'data':'/static/assets/indmarket.mp4'}
    # payload = {'data': 'assets/indmarket.mp4'}
    data = "/static/assets/indmarket.mp4"
    path_details = {}
    path_details = {"vidurl": "/static/assets/indmarket.mp4"}
    # return HttpResponse(json.dumps(payload), content_type='application/json')
    # return JsonResponse({'success':True,'data':'/static/assets/indmarket.mp4'})
    # return JsonResponse(json.loads(json.dumps({'data':'/static/assets/indmarket.mp4'})))
    # return render(request, 'detect/dashboard.html', {"payload": payload})
    # return render(request, 'detect/dashboard.html', {"data": data})
    # return HttpResponse("/static/assets/indmarket.mp4")
    return JsonResponse({'path_details': path_details})
    # return JsonResponse({'success': True, 'data': '/static/assets/indmarket.mp4'})


def maskclick(request, maskval=1):
    # print("sd clicked", maskval)
    path_details = {"vidurl": "/static/assets/FM1591167389.0.mp4"}
    return JsonResponse({'path_details': path_details})


def clicksd(request, fordate='1', cameraid=1):
    #import pdb;pdb.set_trace()
    print("sd clicked 2 para_________----------------------------", fordate, cameraid)
    path_details = {"vidurl": "/static/assets/output.mp4"}
    return JsonResponse({'path_details': path_details})


def clickmask(request, fordate='1', cameraid=1):
    print("mask clicked 2 para%%%%%%%%%%%%%%%%%%%%$$$$$$$$$$$$", fordate, cameraid)
    path_details = {"vidurl": "/static/assets/FM1591167389.0.mp4"}
    return JsonResponse({'path_details': path_details})


def totalSDviolations(request, cameraid=1):
    SDviolations = tbl_Incident_Master.objects.aggregate(Sum('SocialDistance'))
    return JsonResponse({"Social distancing violation": SDviolations})


def totalFMviolations(request, cameraid=1):
    FMviolations = tbl_Incident_Master.objects.aggregate(Sum('FaceMask'))
    return JsonResponse({"Face Mask Violation": FMviolations})

def leftpanel(request):
    menu_list=[]
    locationobjs=LocationMaster.objects.all()

    for location in locationobjs:
        #import pdb;pdb.set_trace()
        print(location.LocationId)
        print(location.location_name)
        elemid='c'+location.LocationId
        index=0
        if location.ParentId==location.LocationId:
            menu_list.append([{'name':location.location_name,'locationid':location.LocationId,'elemid':elemid,'submenus':[]}])
            #menu_list[index]['locationid']=location.LocationId
            index+=1
        elif location.ParentId!=location.LocationId:
            #import pdb;pdb.set_trace()
            print("menu_list", menu_list)
            #menu_list[int(location.ParentId)-1].append({'submenus':location.location_name})
            for menu in menu_list:
                #import pdb;pdb.set_trace()
                elemind='c'+location.LocationId
                if menu[0]['locationid']==location.ParentId:
                    if menu[0]['submenus']:
                        menu[0]['submenus'].append([{'name':location.location_name,'locationid':location.LocationId,'elemid':elemid}])
                    else:
                        menu[0]['submenus'].append([{'name':location.location_name,'locationid':location.LocationId,'elemid':elemid}])
    #import pdb;pdb.set_trace()




    # menu_list.append([{'name':'main gate'}])
    #
    # menu_list[0].append({'submenus': ['camera1']})



    print("menu_list",menu_list)
    return render(request, 'detect/treeviewtemplate.html', {"menu_list": menu_list

                                                      })

def leftcameraclick(request, cameraid=1):
    print("cameraclicked", cameraid)
    request.session['cameraID'] = cameraid
    #violationcount=getviolationscount(request)


    #import pdb;pdb.set_trace()
    #print("request session cam", request.session.get('cameraID'))
    #import pdb;pdb.set_trace()
    # BaseQueryfilter = tbl_Incident_Master.objects.filter(CameraId=cameraid)
    # SDviolations = tbl_Incident_Master.objects.values('CameraId').annotate(sum_SD=Sum('SocialDistance'))
    # FMviolations = tbl_Incident_Master.objects.values('CameraId').annotate(sum_FM=Sum('FaceMask'))

    SDviolations = tbl_Incident_Master.objects.filter(CameraId=cameraid).annotate(sum_SD=Sum('SocialDistance'))
    FMviolations = tbl_Incident_Master.objects.filter(CameraId=cameraid).annotate(sum_FM=Sum('FaceMask'))

    SDviolationsum = 0
    FMviolationsum = 0
    for SDviolation in SDviolations.iterator():
        SDviolationsum += SDviolation.sum_SD

    for FMviolation in FMviolations.iterator():
        FMviolationsum += FMviolation.sum_FM

    # import pdb;pdb.set_trace()
    dateminus30 = (date.today() - timedelta(days=30)).isoformat()
    BaseQueryfilter = tbl_Incident_Master.objects.filter(CameraId=cameraid)
    # import pdb;pdb.set_trace()
    last30SDsum = tbl_Incident_Master.objects.filter(CameraId=cameraid).filter(
        Date_Time_frm__date__gte=dateminus30).aggregate(Sum('SocialDistance'))
    last30FMsum = tbl_Incident_Master.objects.filter(CameraId=cameraid).filter(
        Date_Time_frm__date__gte=dateminus30).aggregate(Sum('FaceMask'))
    print("last 30 sum")
    print(last30SDsum)
    print(last30FMsum)
    last7DaysSD = []
    last7DaysFM = []
    for i in range(1, 8):
        days_before = (date.today() - timedelta(days=i)).isoformat()
        print("days before", days_before)
        lastSD = tbl_Incident_Master.objects.filter(CameraId=cameraid).filter(
            Date_Time_frm__date=days_before).aggregate(Sum('SocialDistance'))

        # if(lastSD['SocialDistance__sum'] is None):
        #     print("in none")
        #     lastSD['SocialDistance__sum']=0
        # import pdb;pdb.set_trace()
        if lastSD['SocialDistance__sum'] == None:
            lastSD['SocialDistance__sum'] = 0
        print("lastSD")
        print(lastSD['SocialDistance__sum'])
        last7DaysSD.append(lastSD['SocialDistance__sum'])

        lastFM = tbl_Incident_Master.objects.filter(CameraId=cameraid).filter(
            Date_Time_frm__date=days_before).aggregate(Sum('FaceMask'))
        # if (lastFM['FaceMask__sum'] is None):
        #     lastFM['FaceMask__sum'] = 0
        if lastFM['FaceMask__sum'] == None:
            lastFM['FaceMask__sum'] = 0
        print("lastFM")
        print(lastFM['FaceMask__sum'])
        last7DaysFM.append(lastFM['FaceMask__sum'])

    last7DaysSD.reverse()
    last7DaysFM.reverse()
    print("last 7 days SD", last7DaysSD)
    print("last 7 days FM", last7DaysFM)
    # import pdb;pdb.set_trace()
    # totalViolationsTop=SDviolations[0].get('sum_SD')+FMviolations[0].get('sum_FM')
    # SDviolationsum=SDviolations[0].get('sum_SD')
    # FMviolationsum=FMviolations[0].get('sum_FM')
    # SDviolationsum = SDviolations[0].sum_SD
    # FMviolationsum = FMviolations[0].sum_FM
    totalViolationsTop = SDviolationsum + FMviolationsum
    # SDviolations={"sumSD":SDviolationsum}
    # FMviolations={"sumFM":FMviolationsum}
    completesums = {}
    completesums['last30SDsum'] = last30SDsum['SocialDistance__sum']
    completesums['last30FMsum'] = last30FMsum['FaceMask__sum']
    completesums['last30Totalsum'] = completesums['last30SDsum'] + completesums['last30FMsum']
    completesums['sumSD'] = SDviolationsum
    completesums['sumFM'] = FMviolationsum
    completesums['sumtotal'] = totalViolationsTop
    completesums['last7DaysSD'] = last7DaysSD
    completesums['last7DaysFM'] = last7DaysFM

    # if type(violationcount) is dict:
    #     print("dicta")
    #     completesums['violations']=violationcount
    return JsonResponse({"completesums": completesums})

    # return render(request, 'detect/dashboard.html', {"SDviolations": SDviolations,
    #                                                  "FMviolations":FMviolations,
    #                                                  "totalViolationsTop":totalViolationsTop})

    # return render(request, 'detect/dashboard.html', {"SDviolations": SDviolations})


    # def livestream(request):

    #     while 1:
    #         print('------------------------------------')
    #         feedlive()
    #     return render(request, 'detect/dashboard.html')