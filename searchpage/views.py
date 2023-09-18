from django.shortcuts import render
from django.core.paginator import Paginator
from django.shortcuts import render
from .traitementpdf import TextPrecessor,SearchEngine
from django.http import FileResponse
from django.conf import settings
import os



instance = TextPrecessor()
my_data = [{"id": ids,"author": author,"titre" : titre,"creator": creator,"subject":subject} for ids, author,titre,creator,subject in zip(                        instance.data["Id"],
                        instance.data["Author"],
                        instance.data["titre"],
                        instance.data["Creator"],
                        instance.data["subject"]
                    )]


def search(request):
    query = request.GET.get('query')
    filter_option = request.GET.get('filter')
    page_obj = None  # Assign an initial value to page_obj
    
    if filter_option == "function1":
        instance = SearchEngine(str(query))
        filtered_data = [d for d in my_data if d["titre"] in instance.tf_idf_similarity()]
        paginator = Paginator(filtered_data, 4)
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)

    elif filter_option == "function2":
        instance = SearchEngine(str(query))
        filtered_data = [d for d in my_data if d["titre"] in instance.complex_search()]
        paginator = Paginator(filtered_data, 4)
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)

    else:
        paginator = Paginator(my_data, 4)
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)
    
    return render(request, "search.html", {'page_obj': page_obj})





