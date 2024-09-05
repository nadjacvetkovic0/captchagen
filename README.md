# Fruit Image Generator

Ovaj projekat je originalno napravljen za generisanje CAPTCHA slika korišćenjem **C-GAN (Composite Generative Adversarial Network)**. Cilj je da se pokaže kako C-GAN generiše nove CAPTCHA slike na osnovu unapred definisanih slika iz baze.

## Pregled projekta

Generisanje CAPTCHA slika postaje sve kompleksnije. U ovom projektu, koristimo C-GAN kako bismo generisali nove CAPTCHA slike, trenirane na bazi slika voća preuzetih sa Kaggle-a.

**Link do baze podataka:**  
https://www.kaggle.com/datasets/moltean/fruits?resource=download

## Testiranje generisanih slika

Za testiranje kvaliteta slika generisanih C-GAN mrežom, koristili smo **ResNet50 CNN** model. Pored toga, ljudsko testiranje je sprovedeno kroz negativni test, čiji link se nalazi ispod.

**Test za ljudsku evaluaciju:**  
https://forms.gle/9Zxu19NgeiU7qFxn9
