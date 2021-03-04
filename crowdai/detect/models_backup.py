from django.db import models



class tbl_Incident_Master(models.Model):
    incidentId = models.CharField('Integer ID',  max_length=40, primary_key=True)
    CameraId = models.CharField('Camera ID', max_length=40)
    Date_Time_frm = models.DateTimeField('Date Time From')
    Date_Time_to=models.DateTimeField('Date Time To')
    VideoPath=models.CharField('Video Path',max_length=100)
    IncidentType=models.CharField('Incident Type',max_length=2)

class tbl_incident_Trans_Mask(models.Model):
    IncidentId_PK=models.CharField('Integer ID',  max_length=40, primary_key=True)
    incidentId=models.ForeignKey(tbl_Incident_Master,on_delete=models.CASCADE)
    ObjectId=models.CharField('Object ID',max_length=40)

class tbl_incident_Trans_SD(models.Model):
    IncidentId_PK=models.CharField('Integer ID',  max_length=40, primary_key=True)
    incidentId=models.ForeignKey(tbl_Incident_Master,on_delete=models.CASCADE)
    ObjectId=models.CharField('Object ID',max_length=40)