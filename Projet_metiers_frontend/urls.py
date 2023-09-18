from django.contrib import admin
from django.urls import path
from landingpage import views
from searchpage import views as vs
from about import views as vs_about 
from statistique import views as vs_stat

urlpatterns = [
    path('admin/', admin.site.urls),
    path("",views.home,name='home'),
    path("upload/",views.upload_files,name='upload'),
    path("search/",vs.search,name='search'),
    path("about/",vs_about.about,name="about"),
    path("stat/",vs_stat.stat,name="stat")
    
]
