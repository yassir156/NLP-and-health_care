from django.shortcuts import render
import os

x = [file for file in os.listdir("templates")]
y = str(len(x))

def home(request):
    context = {'name':"jari amine"}
    return render(request,"index.html",context)


def upload_files(request):
    if request.method == 'POST':
        files = request.FILES.getlist('file')
        for file in files:
            with open("upload_pdfs/"+file.name, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)
        
        return render(request, 'upload.html')
    else:
        return render(request, 'upload.html')
    
def about(request):
    return render(request,"about.html")
