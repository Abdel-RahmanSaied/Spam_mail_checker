from PyQt5 import QtGui, QtWidgets 
from PyQt5.uic import loadUi
import pickle
import qdarkstyle
import sklearn
import sklearn.feature_extraction
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer


vectroizer = pickle.load(open('vectorizer.vect', "rb"))
clf = pickle.load(open("spam classifier.mod" , "rb"))

class SpamClassifier(QtWidgets.QMainWindow):
    def __init__(self):
        super(SpamClassifier , self ).__init__()
        loadUi("main_view.ui" , self )
        self.classify_btn.clicked.connect(self.handle_txt_classify)
        self.show()

    def handle_txt_classify(self):
        email_content = self.email_body_lin.toPlainText()
        result = self.predict(email_content)
        msg = QtWidgets.QMessageBox()
        if result == 'ham' :
            msg.setIconPixmap(QtGui.QPixmap(r"images\icons8-alert-48.png"))
        else : 
            msg.setIconPixmap(QtGui.QPixmap(r"images\icons8-check-64.png"))
        msg.setWindowTitle("Result")
        msg.setText(f"The email classified as {result}")
        msg.exec_()

    def predict(self , email) :
        vec = vectroizer.transform([email]).toarray()
        result = clf.predict(vec)[0]
        return result

app = QtWidgets.QApplication([])

classifier = SpamClassifier()
app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
app.exec_()

#test sec branch kkk
