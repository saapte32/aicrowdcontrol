from django.contrib import admin
from .models import tbl_Incident_Master
from .models import CameraMaster
from .models import LocationMaster
from .forms import LocationMasterForm
from django.contrib.auth.hashers import make_password
from .aes_encrypt import AESCipher

from import_export.admin import ImportExportActionModelAdmin
from import_export import  resources

from django.core.exceptions import ObjectDoesNotExist
from django.core.exceptions import ValidationError
from django.db import IntegrityError





class LocationMasterResource(resources.ModelResource):
    class Meta:
        model = LocationMaster
        skip_unchanged = True
        report_skipped = True
        exclude = ('id',)
        raise_errors=False
        import_id_fields=('LocationId','location_name','Layer','ParentId','camid','IpAdd','locationusername','locationpassword',)

    def before_save_instance(self, instance, using_transactions, dry_run):
        cipher = AESCipher()
        # print("location password")
        # print(instance.locationpassword)

        instance.locationpassword = cipher.encrypt(instance.locationpassword)

        # try:
        #     super(LocationMasterResource, self).save_instance(instance, using_transactions, dry_run=True)
        # except ObjectDoesNotExist:
        #     raise ValidationError("please enter valid camera ID")

    # def save_instance(self,instance, using_transactions=True,dry_run=False):
    #     #import pdb;pdb.set_trace()
    #     try:
    #         super(LocationMasterResource, self).save_instance(instance, using_transactions, dry_run)
    #     except ObjectDoesNotExist as den:
    #         raise Exception('Please enter valid camera ID')
    #     except IntegrityError:
    #         #import pdb;pdb.set_trace()
    #         raise Exception('Please enter valid camera ID')



    # def before_import(self, dataset, using_transactions, dry_run, **kwargs):
    #     try:
    #         super(LocationMasterResource,self).before_import(dataset, using_transactions, dry_run)
    #         # locationmasterresource=LocationMasterResource()
    #         # result = locationmasterresource.import_data(dataset, dry_run=True)
    #
    #     except ObjectDoesNotExist:
    #         print("please enter valid camera ID")












class IncidentAdmin(admin.ModelAdmin):
    list_display = ('CameraId', 'Date_Time_frm', 'SocialDistance', 'FaceMask')
    list_filter = ('CameraId', 'SocialDistance')
    # search_fields = ('Date_Time_frm',)

class CameraAdmin(admin.ModelAdmin):

    list_display = ('Name', 'Focal', 'Range', 'Pixel', 'FPS')
    list_filter = ('Name', 'Range')
    # search_fields = ('Date_Time_frm',)





#class LocationAdmin(admin.ModelAdmin):
class LocationAdmin(ImportExportActionModelAdmin):
    resource_class = LocationMasterResource
    form=LocationMasterForm
    list_display = ('location_name', 'Layer', 'ParentId', 'camid', 'IpAdd','parent_location','locationpassword')
    list_filter = ('location_name', 'Layer', 'camid')
    list_display_links = ('location_name','camid')
    #list_editable = ('Layer','ParentId')

    def parent_location(self,obj):
        #import pdb;pdb.set_trace()
        #refid=int(obj.ParentId)
        parentlocation=LocationMaster.objects.get(LocationId=obj.ParentId)
        print(parentlocation.location_name)
        return "{}".format(parentlocation.location_name)

    def save_model(self, request, instance, form, change):
        instance = form.save(commit=False)

        if not change or not instance.locationpassword:
            cipher = AESCipher()
            instance.locationpassword = cipher.encrypt(instance.locationpassword)
        # instance.modified_by = user
        instance.save()
        form.save_m2m()
        return instance


admin.site.register(tbl_Incident_Master, IncidentAdmin)
admin.site.register(CameraMaster, CameraAdmin)
admin.site.register(LocationMaster, LocationAdmin)




admin.site.site_header = 'AI Crowd Management admin'
admin.site.site_title = 'AI Crowd Management admin'
admin.site.index_title = 'AI Crowd Management administration'