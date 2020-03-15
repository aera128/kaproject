from django import forms


class KAForm(forms.Form):
    raw_text = forms.CharField(widget=forms.Textarea(attrs={'class': 'form-control rounded-0'}), label="Input Text :")
