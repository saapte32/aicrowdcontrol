from django.shortcuts import render
from .models import tbl_Incident_Master
from django.db.models import Sum
from django.http import HttpResponse, JsonResponse
from django.http import Http404
from datetime import date, timedelta
from django.db.models import F, Sum,IntegerField
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage

def index(request):

    return render(request,'detect/dashboard.html')

# Create your views here.

cameraID=1

def violations(request):
        print("violations is called")
        # violations = tbl_Incident_Master.objects.get(pk=question_id)
        BaseQueryfilter = tbl_Incident_Master.objects.filter(CameraId=1)
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

        #violations = tbl_Incident_Master.objects.all()

        day=(date.today() - timedelta(days=0)).isoformat()

        last7DaysSD.reverse()
        last7DaysFM.reverse()
        #import pdb;pdb.set_trace()
        SD = BaseQueryfilter.filter(Date_Time_frm__date=day).annotate(
            totalViolations=Sum(F('SocialDistance') + F('FaceMask'), output_field=IntegerField()))



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





           #maskincident=tbl_incident_Trans_Mask.objects.filter(IncidentId_id=violation.incidentId)
        # for violation in violations:
        #     sdincident=tbl_incident_Trans_SD.objects.filter(IncidentId_id=violation.incidentId)
        #     print(sdincident.ObjectId)

        page_no = request.GET.get('page', 1)
        records_per_page = 10
        paginator = Paginator(SD, records_per_page)
        #import pdb;pdb.set_trace()
        try:
           violationrecords = paginator.page(page_no)
        except PageNotAnInteger:
           violationrecords = paginator.page(1)
        except EmptyPage:
           violationrecords = paginator.page(paginator.num_pages)

        #import pdb;pdb.set_trace()
        return render(request, 'detect/dashboard.html', {"violations": violationrecords,
                                                        "last7DaysSD":last7DaysSD,
                                                        "last7DaysFM":last7DaysFM
                                                        })




    #return render(request,'distancedetector/dashboard.html',{"violations":violations})



    #totalviolations.append(selectedviolations)






def getviolationscount(request):
    print("violation count called too ")
    BaseQueryfilter = tbl_Incident_Master.objects.filter(CameraId=cameraID)
    day = (date.today() - timedelta(days=0)).isoformat()
    SD = BaseQueryfilter.filter(Date_Time_frm__date=day).annotate(
        totalViolations=Sum(F('SocialDistance') + F('FaceMask'), output_field=IntegerField()))
    page_no = request.GET.get('page', 1)
    records_per_page = 10
    paginator = Paginator(SD, records_per_page)

    try:
        violationrecords = paginator.page(page_no)
    except PageNotAnInteger:
        violationrecords = paginator.page(1)
    except EmptyPage:
        violationrecords = paginator.page(paginator.num_pages)


    return render(request, 'detect/violations.html', {"violations": violationrecords

                                                     })




def getFilteredViolations(request):
    print("filtered count called too ")

    BaseQueryfilter = tbl_Incident_Master.objects.filter(CameraId=cameraID)
    day = (date.today() - timedelta(days=0)).isoformat()
    #import pdb; pdb.set_trace()
    filteredrecords=[]
    violationrecords=BaseQueryfilter.filter(Date_Time_frm__date=day).annotate(totalViolations=F('SocialDistance')+F('FaceMask'),incidentdate=F('Date_Time_frm'))

    elemid=0
    for violationrecord in violationrecords:

        if violationrecord.SocialDistance==None:
            violationrecord.SocialDistance=0
        if violationrecord.FaceMask == None:
            violationrecord.FaceMask = 0
        SDcount=violationrecord.SocialDistance
        FMcount=violationrecord.FaceMask
        totalcount=violationrecord.totalViolations
        dictrecord={}
        elemid+=1
        dictrecord['id']=elemid
        dictrecord['SocialDistance']=int(SDcount)
        dictrecord['FaceMask']=int(FMcount)
        dictrecord['Date_Time_frm']=violationrecord.incidentdate
        dictrecord['totalViolations']=totalcount
        filteredrecords.append(dictrecord)
    # if SDViolationToday['SocialDistance__sum'] == None:
    #     SDViolationToday['SocialDistance__sum'] = 0
    # FMViolationToday = BaseQueryfilter.filter(Date_Time_frm__date=day).aggregate(Sum('FaceMask'))
    # if FMViolationToday['FaceMask__sum'] == None:
    #     FMViolationToday['FaceMask__sum'] = 0


    return filteredrecords




def sdclick(request,sdval=1):
    #print("sd clicked",sdval)
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
    #print("sd clicked", maskval)
    path_details = {"vidurl": "/static/assets/indmarket.mp4"}
    return JsonResponse({'path_details': path_details})

def clicksd(request,fordate='1',cameraid=1):
    print("sd clicked 2 para_________----------------------------",fordate,cameraid)
    path_details = {"vidurl": "/static/assets/output.mp4"}
    return JsonResponse({'path_details': path_details})



def clickmask(request,fordate='1',cameraid=1):
    print("mask clicked 2 para%%%%%%%%%%%%%%%%%%%%$$$$$$$$$$$$",fordate,cameraid)
    path_details = {"vidurl": "/static/assets/Output_M.mp4"}
    return JsonResponse({'path_details': path_details})

def totalSDviolations(request,cameraid =1):
    SDviolations = tbl_Incident_Master.objects.aggregate(Sum('SocialDistance') )
    return JsonResponse({"Social distancing violation":SDviolations})

def totalFMviolations(request,cameraid =1):
    FMviolations = tbl_Incident_Master.objects.aggregate(Sum('FaceMask'))
    return JsonResponse({"Face Mask Violation":FMviolations})





def leftcameraclick(request,cameraid =1):
    global cameraID
    cameraID=cameraid
    print("cameraclicked",cameraid)
    # import pdb;pdb.set_trace()
    #BaseQueryfilter = tbl_Incident_Master.objects.filter(CameraId=cameraid)
    # SDviolations = tbl_Incident_Master.objects.values('CameraId').annotate(sum_SD=Sum('SocialDistance'))
    # FMviolations = tbl_Incident_Master.objects.values('CameraId').annotate(sum_FM=Sum('FaceMask'))

    SDviolations = tbl_Incident_Master.objects.filter(CameraId=cameraid).annotate(sum_SD=Sum('SocialDistance'))
    FMviolations = tbl_Incident_Master.objects.filter(CameraId=cameraid).annotate(sum_FM=Sum('FaceMask'))

    SDviolationsum=0
    FMviolationsum=0
    for SDviolation in SDviolations.iterator():
        SDviolationsum += SDviolation.sum_SD

    for FMviolation in FMviolations.iterator():
        FMviolationsum += FMviolation.sum_FM


    #import pdb;pdb.set_trace()
    dateminus30=(date.today()-timedelta(days=30)).isoformat()
    BaseQueryfilter=tbl_Incident_Master.objects.filter(CameraId=cameraid)
    #import pdb;pdb.set_trace()
    last30SDsum=tbl_Incident_Master.objects.filter(CameraId=cameraid).filter(Date_Time_frm__date__gte=dateminus30).aggregate(Sum('SocialDistance'))
    last30FMsum = tbl_Incident_Master.objects.filter(CameraId=cameraid).filter(Date_Time_frm__date__gte=dateminus30).aggregate(Sum('FaceMask'))
    print("last 30 sum")
    print(last30SDsum)
    print(last30FMsum)
    last7DaysSD=[]
    last7DaysFM=[]
    for i in range(1, 8):
        days_before = (date.today() - timedelta(days=i)).isoformat()
        print("days before", days_before)
        lastSD = tbl_Incident_Master.objects.filter(CameraId=cameraid).filter(Date_Time_frm__date=days_before).aggregate(Sum('SocialDistance'))

        # if(lastSD['SocialDistance__sum'] is None):
        #     print("in none")
        #     lastSD['SocialDistance__sum']=0
        #import pdb;pdb.set_trace()
        if lastSD['SocialDistance__sum'] == None:
            lastSD['SocialDistance__sum']=0
        print("lastSD")
        print(lastSD['SocialDistance__sum'])
        last7DaysSD.append(lastSD['SocialDistance__sum'])

        lastFM = tbl_Incident_Master.objects.filter(CameraId=cameraid).filter(Date_Time_frm__date=days_before).aggregate(Sum('FaceMask'))
        # if (lastFM['FaceMask__sum'] is None):
        #     lastFM['FaceMask__sum'] = 0
        if lastFM['FaceMask__sum'] == None:
            lastFM['FaceMask__sum'] = 0
        print("lastFM")
        print(lastFM['FaceMask__sum'])
        last7DaysFM.append(lastFM['FaceMask__sum'])

    last7DaysSD.reverse()
    last7DaysFM.reverse()
    print("last 7 days SD",last7DaysSD)
    print("last 7 days FM",last7DaysFM)
    #import pdb;pdb.set_trace()
    #totalViolationsTop=SDviolations[0].get('sum_SD')+FMviolations[0].get('sum_FM')
    # SDviolationsum=SDviolations[0].get('sum_SD')
    # FMviolationsum=FMviolations[0].get('sum_FM')
    #SDviolationsum = SDviolations[0].sum_SD
    #FMviolationsum = FMviolations[0].sum_FM
    totalViolationsTop=SDviolationsum+FMviolationsum
    # SDviolations={"sumSD":SDviolationsum}
    # FMviolations={"sumFM":FMviolationsum}
    completesums={}
    completesums['last30SDsum']=last30SDsum['SocialDistance__sum']
    completesums['last30FMsum'] = last30FMsum['FaceMask__sum']
    completesums['last30Totalsum']= completesums['last30SDsum']+ completesums['last30FMsum']
    completesums['sumSD']=SDviolationsum
    completesums['sumFM']=FMviolationsum
    completesums['sumtotal'] = totalViolationsTop
    completesums['last7DaysSD']=last7DaysSD
    completesums['last7DaysFM']=last7DaysFM
    #completesums['violations']=getFilteredViolations(request)
    return JsonResponse({"completesums":completesums})

    # return render(request, 'detect/dashboard.html', {"SDviolations": SDviolations,
    #                                                  "FMviolations":FMviolations,
    #                                                  "totalViolationsTop":totalViolationsTop})

   # return render(request, 'detect/dashboard.html', {"SDviolations": SDviolations})






def resetviolations(request,cameraid):
        print("reset violations is called camera id is ", cameraid)
        # violations = tbl_Incident_Master.objects.get(pk=question_id)

        BaseQueryfilter = tbl_Incident_Master.objects.filter(CameraId=cameraid)


        #violations = tbl_Incident_Master.objects.all()

        day=(date.today() - timedelta(days=0)).isoformat()


        #import pdb;pdb.set_trace()
        SD = BaseQueryfilter.filter(Date_Time_frm__date=day).annotate(
            totalViolations=Sum(F('SocialDistance') + F('FaceMask'), output_field=IntegerField()))




        page_no = request.GET.get('page', 1)
        records_per_page = 10
        paginator = Paginator(SD, records_per_page)
        #import pdb;pdb.set_trace()
        try:
           violationrecords = paginator.page(page_no)
        except PageNotAnInteger:
           violationrecords = paginator.page(1)
        except EmptyPage:
           violationrecords = paginator.page(paginator.num_pages)

        #import pdb;pdb.set_trace()
        return render(request, 'detect/violations.html', {"violations": violationrecords

                                                        })


