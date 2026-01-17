import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle
import warnings
warnings.filterwarnings('ignore')

class CDTS_Model:
    """
    CDTS model - TF-IDF + SVM
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents='unicode',
            lowercase=True,
            token_pattern=r'\b\w+\b'
        )
        
        self.classifier = LinearSVC(
            C=1.0,
            class_weight='balanced',
            max_iter=10000,
            random_state=42,
            dual=False
        )
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'of', 'for', 'with', 'as', 'by'}
        
    def preprocess_text(self, text):
        """
        remove symbols, links, stop words
        convert to lowercase
        tokenization
        stemming/lemmatization
        """
        if pd.isna(text) or str(text).strip() == '':
            return ""
        
        text = str(text).lower()
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        text = re.sub(r'[^a-z\s\']', ' ', text)
        
        contractions = {
            "don't": "do not", "doesn't": "does not", "didn't": "did not",
            "can't": "cannot", "couldn't": "could not", "won't": "will not",
            "wouldn't": "would not", "shouldn't": "should not",
            "isn't": "is not", "aren't": "are not", "wasn't": "was not",
            "weren't": "were not", "haven't": "have not", "hasn't": "has not",
            "i'm": "i am", "i've": "i have", "i'll": "i will"
        }
        
        for cont, exp in contractions.items():
            text = text.replace(cont, exp)
        
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        processed = []
        for token in tokens:
            if len(token) > 1 and token not in self.stop_words:
                lemma = self.lemmatizer.lemmatize(token, pos='v')
                processed.append(lemma)
        
        return ' '.join(processed)
    
    def train(self, X_train, y_train):
        """train the model"""
        print("training model")
        
        print("\npreprocessing texts")
        X_processed = [self.preprocess_text(text) for text in X_train]
        
        print("extracting TF-IDF features")
        X_tfidf = self.vectorizer.fit_transform(X_processed)
        print(f"  feature matrix: {X_tfidf.shape}")
        
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\ntraining set class distribution:")
        for label, count in zip(unique, counts):
            print(f"  {label}: {count} ({count/len(y_train)*100:.1f}%)")
        
        print("\ntraining SVM classifier")
        self.classifier.fit(X_tfidf, y_train)
        
        print("\ncross-validation")
        cv_scores = cross_val_score(self.classifier, X_tfidf, y_train, 
                                    cv=min(5, len(unique)), scoring='f1_weighted')
        print(f"  CV f1-scores: {[f'{s:.3f}' for s in cv_scores]}")
        print(f"  mean CV f1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        print("\ntraining completed")
    
    def predict(self, X_test):
        """predict"""
        X_processed = [self.preprocess_text(text) for text in X_test]
        X_tfidf = self.vectorizer.transform(X_processed)
        return self.classifier.predict(X_tfidf)
    
    def evaluate(self, X_test, y_test):
        """evaluate model"""
        print("EVALUATION RESULTS")
        
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\n{'performance metrics':^70}")
        print("-"*70)
        print(f"{'precision:':<20} {precision:.4f} ({precision*100:.2f}%)")
        print(f"{'recall:':<20} {recall:.4f} ({recall*100:.2f}%)")
        print(f"{'f1-score:':<20} {f1:.4f} ({f1*100:.2f}%)")
        print(f"{'accuracy:':<20} {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("-"*70)
        
        print(f"\n{'comparison with paper':^70}")
        print("-"*70)
        print(f"{'metric':<20} {'my result':>15} {'paper result':>15} {'difference':>15}")
        print("-"*70)
        print(f"{'precision':<20} {precision*100:>14.2f}% {95:>14.2f}% {(precision*100-95):>+14.2f}%")
        print(f"{'recall':<20} {recall*100:>14.2f}% {94:>14.2f}% {(recall*100-94):>+14.2f}%")
        print(f"{'f1-score':<20} {f1*100:>14.2f}% {94:>14.2f}% {(f1*100-94):>+14.2f}%")
        print(f"{'accuracy':<20} {accuracy*100:>14.2f}% {94:>14.2f}% {(accuracy*100-94):>+14.2f}%")
        print("-"*70)
        
        print(f"\n{'detailed classification report':^70}")
        print("-"*70)
        print(classification_report(y_pred, y_test, zero_division=0))
        
        print(f"\n{'confusion matrix':^70}")
        print("-"*70)
        cm = confusion_matrix(y_test, y_pred)
        print(f"correct predictions: {np.trace(cm)} / {len(y_test)}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def save_model(self, filepath='cdts_model.pkl'):
        """save model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier
            }, f)
        print(f"\nmodel saved to {filepath}")


def load_dataset(filepath):
    """
    load the CBT dataset'
    """
    print("LOADING DATASET")
    
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(filepath, encoding='latin-1')
        except:
            df = pd.read_csv(filepath, encoding='iso-8859-1')
    
    print(f"\ndataset: {filepath}")
    print(f"total samples: {len(df)}")
    print(f"columns: {df.columns.tolist()}")
    
    texts = df['negative_thought'].tolist()
    

    labels = []
    for distortion in df['distortions'].tolist():
        if pd.notna(distortion):
            clean_label = str(distortion).split(',')[0].strip()
            
            labels.append(clean_label)
    
    print(f"\n{'cognitive distortion distribution':^70}")
    print("-"*70)
    label_counts = pd.Series(labels).value_counts()
    print(label_counts)
    
    print(f"\n{'distribution percentages':^70}")
    print("-"*70)
    for distortion, count in label_counts.items():
        pct = count/len(labels)*100
        print(f"{distortion}: {count} ({pct:.2f}%)")
    
    return texts, labels, df


def main():
    """main execution"""
    print("CDTS MODEL - COGNITIVE DISTORTION DETECTION")
    
    dataset_path = 'cbt_df.csv'
    
    try:
        texts, labels, df = load_dataset(dataset_path)
    except FileNotFoundError:
        print(f"\n error: '{dataset_path}' not found")
        return
    except Exception as e:
        print(f"\n error loading dataset {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("DATA SPLIT (80% train / 20% test)")
    
    label_counts = pd.Series(labels).value_counts()
    min_count = label_counts.min()
    
    if min_count >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels,
            test_size=0.2,
            random_state=42
        )
    
    print(f"training set: {len(X_train)} samples ({len(X_train)/len(texts)*100:.1f}%)")
    print(f"testing set: {len(X_test)} samples ({len(X_test)/len(texts)*100:.1f}%)")
    
    model = CDTS_Model()
    model.train(X_train, y_train)
    
    metrics = model.evaluate(X_test, y_test)
    
    print("FINAL RESULTS SUMMARY")
    print(f"\ndataset: {len(texts)} samples")
    print(f"number of distortion types: {len(label_counts)}")
    print(f"\nperformance:")
    print(f"  accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  precision: {metrics['precision']*100:.2f}%")
    print(f"  recall:    {metrics['recall']*100:.2f}%")
    print(f"  f1-score:  {metrics['f1_score']*100:.2f}%")
    
    print("\n" + "-"*70)
    print("PAPER TARGET:")
    print("  precision: 95%")
    print("  recall:    94%") 
    print("  f1-score:  94%")
    print("  accuracy:  94%")
    print("-"*70)
    
    save_choice = input("\nsave model? (y/n): ").lower()
    if save_choice == 'y':
        model.save_model('cdts_model.pkl')
    
    test_choice = input("\ntest with custom examples? (y/n): ").lower()
    if test_choice == 'y':
        print("TESTING EXAMPLES")
        
        examples = [
            "I failed the test, I'm a complete failure",
            "Nobody ever listens to me",
            "They must think I'm stupid",
            "If I don't get this job, my life is over",
            "I feel like a loser, so I must be one"
        ]
        
        for text in examples:
            pred = model.predict([text])[0]
            print(f"\ntext: '{text}'")
            print(f"predicted: {pred}")
        
        while True:
            custom = input("\nenter text (or 'quit'): ")
            if custom.lower() == 'quit':
                break
            pred = model.predict([custom])[0]
            print(f" predicted: {pred}")
    
    print("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()