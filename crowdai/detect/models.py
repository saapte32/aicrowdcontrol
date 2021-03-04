from django.db import models
from django.contrib import admin
from django import forms

from django.core.validators import RegexValidator

alphanumeric = RegexValidator(r'^[0-9a-zA-Z]*$', 'Only alphanumeric characters are allowed.')

numeric = RegexValidator(r'^[0-9]\d*(\.\d+)*$', 'Only numeric characters are allowed.')

alphanumericwspace = RegexValidator(r'^[0-9a-zA-Z\s]*$', 'Only alphanumeric characters and space are allowed.')

class CameraMaster(models.Model):
    camId = models.CharField('ID', max_length=50,validators=[alphanumeric] )
    Name = models.CharField('Name', max_length=100, primary_key=True,  validators=[alphanumericwspace])
    Focal = models.FloatField('Focal Length', max_length=40,validators=[numeric])
    Range = models.FloatField('Max Range of Camera', max_length=40,validators=[numeric])
    Pixel = models.FloatField('Pixel width', max_length=40,validators=[numeric])
    FPS = models.FloatField('FramePerSecond(low fps = slower speed)', max_length=40,validators=[numeric])
    relay=models.CharField('Relay', max_length=100, validators=[alphanumeric])
    def __str__(self):
        return self.Name


class NameChoiceField(forms.ModelChoiceField):
    def label_from_instance(self, obj):
        return "{}".format(obj.Name)

    def formfield_for_foreignkey(self, db_field, request, **kwargs):
        if db_field.name == 'Name':
            return NameChoiceField(queryset=CameraMaster.objects.all())
        return super().formfield_for_foreignkey(db_field, request, **kwargs)


class LocationMaster(models.Model):
    LocationId = models.CharField('Location ID', max_length=40, primary_key=True,  validators=[alphanumeric])
    location_name = models.CharField('Location Name', max_length=40,  validators=[alphanumericwspace])
    Layer = models.CharField('Layer Number', max_length=40,  validators=[alphanumeric])
    ParentId = models.CharField('Parent ID', max_length=40,  validators=[alphanumeric])
    camid = models.ForeignKey(CameraMaster, on_delete=models.DO_NOTHING)
    # camid = NameDrop(queryset=CameraMaster.objects.select_related('Name').all())
    IpAdd = models.CharField('Ip Address', max_length=40,  validators=[alphanumeric])
    locationusername = models.CharField('Location Username', max_length=50, validators=[alphanumeric])
    locationpassword = models.TextField('Location Password', max_length=500 )
    bottomleft = models.CharField('Bottom Left', max_length=40,  validators=[numeric])
    bottomright = models.CharField('Bottom Right', max_length=40,  validators=[numeric])
    topleft = models.CharField('Top Left', max_length=40, validators=[numeric])
    topright = models.CharField('Top Right', max_length=40, validators=[numeric])
    vertical = models.CharField('Vertical', max_length=40, validators=[numeric])
    mid = models.CharField('mid', max_length=40, validators=[numeric])
    horizontal = models.CharField('horizontal', max_length=40, validators=[numeric])







class tbl_Incident_Master(models.Model):
    CameraId = models.CharField('Camera ID', max_length=40, validators=[alphanumeric])
    Date_Time_frm = models.DateTimeField('Date Time From')
    Date_Time_to=models.DateTimeField('Date Time To')
    SDVideoPath=models.CharField('Video Path',max_length=100 )
    FMVideoPath = models.CharField('Video Path', max_length=100)
    SocialDistance=models.CharField('Social Distance',max_length=40, validators=[alphanumeric])
    FaceMask = models.CharField('Face Mask', max_length=40, validators=[alphanumeric])



#IncidentType=models.CharField('Incident Type',max_length=2)

# class tbl_incident_Trans_Mask(models.Model):
#     IncidentId_PK=models.CharField('Integer ID',  max_length=40, primary_key=True)
#     incidentId=models.ForeignKey(tbl_Incident_Master,on_delete=models.CASCADE)
#     ObjectId=models.CharField('Object ID',max_length=40)
#
# class tbl_incident_Trans_SD(models.Model):
#     IncidentId_PK=models.CharField('Integer ID',  max_length=40, primary_key=True)
#     incidentId=models.ForeignKey(tbl_Incident_Master,on_delete=models.CASCADE)
#     ObjectId=models.CharField('Object ID',max_length=40)
