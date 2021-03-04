

from .models import *
from django.forms import ModelForm, PasswordInput
from django.contrib.auth.models import AbstractUser
from passlib.hash  import pbkdf2_sha256
from django.contrib.auth.hashers import make_password

class LocationMasterForm(ModelForm):
    #camerapassword = forms.CharField(widget=PasswordInput())


    #hashcamerapassword= pbkdf2_sha256.hash(str(camerapassword))
    # print(pbkdf2_sha256.verify(str(camerapassword), hashcamerapassword))
    # print(pbkdf2_sha256.verify("rock", hashcamerapassword))


    class Meta:
        model = LocationMaster
        fields= '__all__'
        widgets = {
            'locationpassword': PasswordInput(),
        }


    # def save(self, commit=True):
    #     user = super(CameraMasterForm, self).save(commit=False)
    #     user.set_password(self.cleaned_data["camerapassword"])
    #     if commit:
    #         user.save()
    #     return user

    # def save(self, *args, **kwargs):
    #     form=CameraMasterForm(commit=False)
    #     form.camerapassword = make_password(camerapassword)
    #     super(CameraMasterForm, self).save(*args, **kwargs)

    # def save(self):
    #     data = self.cleaned_data
    #     self.camerapassword = make_password(data['camerapassword'])
    #     # user = User(email=data['email'], first_name=data['first_name'],
    #     #             last_name=data['last_name'], password1=data['password1'],
    #     #             password2=data['password2'])
    #     # user.save()
    #
    #     super.save(commit=False)
    #     super(CameraMasterForm, self).save(*args, **kwargs)
