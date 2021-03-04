from django.urls import path

from . import views
from django.conf import settings
from django.conf.urls.static import static

from django.views.static import serve

urlpatterns = [
    # ex: /polls/
    path('', views.violations, name='violations'),
    # path('sdetect/<int:sdval>/', views.sdclick,name='sdclick'),
    # path('maskdetect/<int:maskval>/', views.maskclick,name='maskclick'),
    path('sdetect/<str:fordate>/<int:cameraid>', views.clicksd,name='clicksd'),
    path('maskdetect/<str:fordate>/<int:cameraid>', views.clickmask,name='clickmask'),
    path('leftcameraclick/<str:cameraid>', views.leftcameraclick, name='leftcameraclick'),
    path('getviolationscount/', views.getviolationscount, name='getviolationscount'),
    path('leftpanel/', views.leftpanel, name='leftpanel'),
    #path('resetviolations/<str:cameraid>', views.resetviolations, name='resetviolations'),
    # path('<int:question_id>/', views.detail, name='detail'),
    #
    # path('<int:question_id>/results/', views.results, name='results'),
    #
    # path('<int:question_id>/vote/', views.vote, name='vote'),
    #path(r'^static/(?P<path>.*)$', serve,{'document_root': settings.STATIC_ROOT}),
    #path(r'^static/(?P<path>.*)$', serve, {'document_root': settings.STATIC_ROOT})
]



