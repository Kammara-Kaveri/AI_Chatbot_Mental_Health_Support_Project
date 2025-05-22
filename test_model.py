import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress tokenizers parallelism warning

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from better_profanity import profanity
import re
from spellchecker import SpellChecker
import language_tool_python

# Initialize grammar and spelling checkers
spell_checker = SpellChecker()
tool = language_tool_python.LanguageTool('en-US')

def correct_grammar_and_spelling(text):
    words = text.split()
    corrected_words = []
    for word in words:
        if re.match(r'^\W+$', word):
            corrected_words.append(word)
        else:
            corrected_word = spell_checker.correction(word)
            corrected_words.append(corrected_word if corrected_word else word)
    corrected_text = ' '.join(corrected_words)
    matches = tool.check(corrected_text)
    corrected_text = language_tool_python.utils.correct(corrected_text, matches)
    return corrected_text

profanity.load_censor_words()
def filter_offensive_input(user_input):
    clinical_terms = [
        'pain', 'anxiety', 'trauma', 'ptsd', 'Anxiety', 'Depression', 'Bipolar disorder',
        'PTSD (Post-Traumatic Stress Disorder)', 'OCD (Obsessive-Compulsive Disorder)',
        'Schizophrenia', 'Panic attacks', 'Insomnia', 'Social anxiety', 'Sadness',
        'Loneliness', 'Hopelessness', 'Fear', 'Guilt', 'Fatigue', 'Mood swings',
        'Loss of appetite', 'Restlessness', 'Crying spells', 'Trouble concentrating',
        'Mindfulness', 'Meditation', 'Deep breathing', 'Journaling', 'Self-care',
        'Therapy', 'Counseling', 'Support groups', 'Crisis hotline', 'Self-harm',
        'Suicidal thoughts', 'Suicide prevention', 'Mental breakdown', 'Urgent help',
        'Helpline', 'Emergency contact', 'Immediate danger', 'Diagnosis', 'Prescription',
        'Medication', 'Side effects', 'Therapy plan', 'Mental health professional',
        'Psychiatrist', 'Psychologist', 'Treatment', 'Workload', 'Work-life', 'Friendship'
    ]
    censored = profanity.censor(user_input)
    if censored != user_input:
        for term in clinical_terms:
            censored = censored.replace(profanity.censor(term), term)
        if censored != user_input:
            return True, "Input contains inappropriate language. Please rephrase."
    return False, "Input is safe."

def clean_response(response, user_input):
    response = response.strip()
    response = re.sub(r'^\.+|\.+$|[.,;]\s*[.,;]', '', response)
    response = re.sub(r'\s+', ' ', response).strip()
    sentences = [s.strip() for s in response.split('.') if s.strip()]
    response = '. '.join(sentences[:2]) + ('.' if sentences else '')
    if (len(response) < 25 or
        len(set(response.split())) < 10 or
        not re.search(r'[a-zA-Z]{3,}', response) or
        not re.search(r'\b(am|is|are|was|were|be|being|been|have|has|had|do|does|did|feel|think|say)\b', response, re.IGNORECASE) or
        re.search(r'\b(you|to|the|I)\b.*\b(you|to|the|I)\b.*\b(you|to|the|I)\b', response, re.IGNORECASE) or
        len(sentences) < 2 or
        any(len(s.split()) < 5 for s in sentences)):
        return None
    return response

def extract_emotion(user_input):
    user_input_lower = user_input.lower()
    # Emotional Scenarios
    if 'angry' in user_input_lower:
        return 'angry'
    elif 'frustrated' in user_input_lower:
        return 'frustrated'
    elif 'overwhelmed' in user_input_lower:
        return 'overwhelmed'
    elif 'stressed' in user_input_lower:
        return 'stressed'
    elif 'anxious' in user_input_lower:
        return 'anxious'
    elif 'scared' in user_input_lower or 'afraid' in user_input_lower:
        return 'scared'
    elif 'terrified' in user_input_lower:
        return 'terrified'
    elif 'nervous' in user_input_lower:
        return 'nervous'
    elif 'worried' in user_input_lower:
        return 'worried'
    elif 'panicked' in user_input_lower or 'panic attack' in user_input_lower:
        return 'panicked'
    elif 'sad' in user_input_lower:
        return 'sad'
    elif 'depressed' in user_input_lower:
        return 'depressed'
    elif 'hopeless' in user_input_lower:
        return 'hopeless'
    elif 'lonely' in user_input_lower:
        return 'lonely'
    elif 'isolated' in user_input_lower:
        return 'isolated'
    elif 'died' in user_input_lower or 'loss' in user_input_lower or 'passed away' in user_input_lower:
        return 'grieving'
    elif 'heartbroken' in user_input_lower:
        return 'heartbroken'
    elif 'guilty' in user_input_lower:
        return 'guilty'
    elif 'ashamed' in user_input_lower:
        return 'ashamed'
    elif 'embarrassed' in user_input_lower:
        return 'embarrassed'
    elif 'jealous' in user_input_lower:
        return 'jealous'
    elif 'insecure' in user_input_lower:
        return 'insecure'
    elif 'confused' in user_input_lower:
        return 'confused'
    elif 'lost' in user_input_lower:
        return 'lost'
    elif 'empty' in user_input_lower:
        return 'empty'
    elif 'numb' in user_input_lower:
        return 'numb'
    elif 'exhausted' in user_input_lower or 'fatigue' in user_input_lower:
        return 'exhausted'
    elif 'burned out' in user_input_lower:
        return 'burned out'
    elif 'unmotivated' in user_input_lower:
        return 'unmotivated'
    elif 'restless' in user_input_lower:
        return 'restless'
    # Mental Health Conditions and Symptoms
    elif 'anxiety' in user_input_lower:
        return 'anxiety'
    elif 'depression' in user_input_lower:
        return 'depression'
    elif 'ptsd' in user_input_lower or 'trauma' in user_input_lower:
        return 'ptsd'
    elif 'ocd' in user_input_lower or 'obsessive' in user_input_lower:
        return 'ocd'
    elif 'bipolar' in user_input_lower or 'mood swings' in user_input_lower:
        return 'bipolar'
    elif 'schizophrenia' in user_input_lower:
        return 'schizophrenia'
    elif 'panic attacks' in user_input_lower:
        return 'panic attacks'
    elif 'social anxiety' in user_input_lower:
        return 'social anxiety'
    elif 'sleep' in user_input_lower or 'insomnia' in user_input_lower:
        return 'having trouble sleeping'
    elif 'loss of appetite' in user_input_lower:
        return 'loss of appetite'
    elif 'crying spells' in user_input_lower:
        return 'crying spells'
    elif 'trouble concentrating' in user_input_lower:
        return 'trouble concentrating'
    elif 'self-harm' in user_input_lower:
        return 'self-harm'
    elif 'suicidal thoughts' in user_input_lower or 'suicide' in user_input_lower:
        return 'suicidal thoughts'
    elif 'mental breakdown' in user_input_lower:
        return 'mental breakdown'
    elif 'overthinking' in user_input_lower:
        return 'overthinking'
    elif 'intrusive thoughts' in user_input_lower:
        return 'intrusive thoughts'
    elif 'dissociation' in user_input_lower:
        return 'dissociation'
    elif 'hyperventilating' in user_input_lower:
        return 'hyperventilating'
    elif 'flashbacks' in user_input_lower:
        return 'flashbacks'
    # Relationship and Social Scenarios
    elif 'friendship' in user_input_lower or 'friend' in user_input_lower:
        return 'anxious about friendship'
    elif 'friends don’t value me' in user_input_lower or 'not valued' in user_input_lower:
        return 'friends don’t value me'
    elif 'cheating' in user_input_lower:
        return 'cheating'
    elif 'breakup' in user_input_lower or 'broke up' in user_input_lower:
        return 'breakup'
    elif 'divorce' in user_input_lower or 'separated' in user_input_lower:
        return 'divorce'
    elif 'lonely in relationship' in user_input_lower:
        return 'lonely in relationship'
    elif 'family issues' in user_input_lower or 'family problems' in user_input_lower:
        return 'family issues'
    elif 'parenting stress' in user_input_lower:
        return 'parenting stress'
    elif 'workplace conflict' in user_input_lower:
        return 'workplace conflict'
    elif 'bullying' in user_input_lower:
        return 'bullying'
    # Work and Lifestyle Scenarios
    elif 'workload' in user_input_lower or 'work-life' in user_input_lower:
        return 'overwhelmed by workload'
    elif 'job stress' in user_input_lower:
        return 'job stress'
    elif 'unemployment' in user_input_lower or 'lost job' in user_input_lower:
        return 'unemployment'
    elif 'financial stress' in user_input_lower or 'money problems' in user_input_lower:
        return 'financial stress'
    elif 'daily' in user_input_lower and 'routine' in user_input_lower:
        return 'seeking daily routine'
    elif 'procrastination' in user_input_lower:
        return 'procrastination'
    elif 'time management' in user_input_lower:
        return 'time management'
    elif 'lack of energy' in user_input_lower:
        return 'lack of energy'
    elif 'exercise motivation' in user_input_lower:
        return 'exercise motivation'
    elif 'healthy eating' in user_input_lower:
        return 'healthy eating'
    # Physical Health and Self-Care
    elif 'pain' in user_input_lower:
        return 'in pain'
    elif 'headache' in user_input_lower:
        return 'headache'
    elif 'stomach pain' in user_input_lower:
        return 'stomach pain'
    elif 'chronic pain' in user_input_lower:
        return 'chronic pain'
    elif 'self-care' in user_input_lower:
        return 'self-care'
    elif 'mindfulness' in user_input_lower:
        return 'mindfulness'
    elif 'meditation' in user_input_lower:
        return 'meditation'
    elif 'deep breathing' in user_input_lower:
        return 'deep breathing'
    elif 'journaling' in user_input_lower:
        return 'journaling'
    # Seeking Professional Help
    elif 'therapy' in user_input_lower or 'therapist' in user_input_lower:
        return 'considering therapy'
    elif 'counseling' in user_input_lower:
        return 'counseling'
    elif 'support groups' in user_input_lower:
        return 'support groups'
    elif 'crisis hotline' in user_input_lower or 'urgent help' in user_input_lower:
        return 'crisis hotline'
    elif 'helpline' in user_input_lower:
        return 'helpline'
    elif 'emergency contact' in user_input_lower or 'immediate danger' in user_input_lower:
        return 'emergency contact'
    elif 'psychiatrist' in user_input_lower:
        return 'psychiatrist'
    elif 'psychologist' in user_input_lower:
        return 'psychologist'
    elif 'diagnosis' in user_input_lower:
        return 'diagnosis'
    elif 'prescription' in user_input_lower or 'medication' in user_input_lower:
        return 'prescription'
    # Other Mental Health Support Scenarios
    elif 'relapse' in user_input_lower:
        return 'relapse'
    elif 'therapy plan' in user_input_lower:
        return 'therapy plan'
    elif 'side effects' in user_input_lower:
        return 'side effects'
    elif 'treatment' in user_input_lower:
        return 'treatment'
    elif 'mental health professional' in user_input_lower:
        return 'mental health professional'
    elif 'coping strategies' in user_input_lower:
        return 'coping strategies'
    elif 'stress management' in user_input_lower:
        return 'stress management'
    elif 'emotional regulation' in user_input_lower:
        return 'emotional regulation'
    elif 'self-esteem' in user_input_lower:
        return 'self-esteem'
    elif 'motivation' in user_input_lower:
        return 'motivation'
    # Additional Scenarios
    elif 'fear of failure' in user_input_lower:
        return 'fear of failure'
    elif 'perfectionism' in user_input_lower:
        return 'perfectionism'
    elif 'grief anniversary' in user_input_lower:
        return 'grief anniversary'
    elif 'seasonal depression' in user_input_lower:
        return 'seasonal depression'
    elif 'identity crisis' in user_input_lower:
        return 'identity crisis'
    elif 'trust issues' in user_input_lower:
        return 'trust issues'
    elif 'abandonment fears' in user_input_lower:
        return 'abandonment fears'
    elif 'codependency' in user_input_lower:
        return 'codependency'
    elif 'anger management' in user_input_lower:
        return 'anger management'
    elif 'self-compassion' in user_input_lower:
        return 'self-compassion'
    # New Scenarios 
    elif 'disappointed' in user_input_lower:
        return 'disappointed'  
    elif 'regret' in user_input_lower:
        return 'regretful' 
    elif 'shocked' in user_input_lower:
        return 'shocked'  
    elif 'disgusted' in user_input_lower:
        return 'disgusted'  
    elif 'irritated' in user_input_lower:
        return 'irritated'  
    elif 'bored' in user_input_lower:
        return 'bored'  
    elif 'apathetic' in user_input_lower:
        return 'apathetic'  
    elif 'helpless' in user_input_lower:
        return 'helpless'  
    elif 'powerless' in user_input_lower:
        return 'powerless'  
    elif 'trapped' in user_input_lower:
        return 'trapped'  
    elif 'resentful' in user_input_lower:
        return 'resentful'  
    elif 'bitter' in user_input_lower:
        return 'bitter'  
    elif 'envious' in user_input_lower:
        return 'envious'  
    elif 'doubtful' in user_input_lower:
        return 'doubtful'  
    elif 'skeptical' in user_input_lower:
        return 'skeptical'  
    elif 'paranoid' in user_input_lower:
        return 'paranoid'  
    elif 'claustrophobic' in user_input_lower:
        return 'claustrophobic'  
    elif 'phobia' in user_input_lower:
        return 'phobia'  
    elif 'agitated' in user_input_lower:
        return 'agitated' 
    elif 'tense' in user_input_lower:
        return 'tense'  
    elif 'distracted' in user_input_lower:
        return 'distracted'  
    elif 'disoriented' in user_input_lower:
        return 'disoriented'  
    elif 'alienated' in user_input_lower:
        return 'alienated'  
    elif 'misunderstood' in user_input_lower:
        return 'misunderstood'  
    elif 'judged' in user_input_lower:
        return 'judged'  
    elif 'excluded' in user_input_lower:
        return 'excluded'  
    elif 'rejected' in user_input_lower:
        return 'rejected'  
    elif 'betrayed' in user_input_lower:
        return 'betrayed'  
    elif 'abused' in user_input_lower:
        return 'abused' 
    elif 'neglected' in user_input_lower:
        return 'neglected'  
    elif 'unsafe' in user_input_lower:
        return 'unsafe'  
    elif 'vulnerable' in user_input_lower:
        return 'vulnerable'  
    elif 'exposed' in user_input_lower:
        return 'exposed'  
    elif 'overstimulated' in user_input_lower:
        return 'overstimulated'  
    elif 'sensory overload' in user_input_lower:
        return 'sensory overload'  
    elif 'addiction' in user_input_lower:
        return 'addiction'  
    elif 'withdrawal' in user_input_lower:
        return 'withdrawal'  
    elif 'eating disorder' in user_input_lower:
        return 'eating disorder'  
    elif 'anorexia' in user_input_lower:
        return 'anorexia'  
    elif 'bulimia' in user_input_lower:
        return 'bulimia'  
    elif 'binge eating' in user_input_lower:
        return 'binge eating'  
    elif 'body dysmorphia' in user_input_lower:
        return 'body dysmorphia'  
    elif 'gender dysphoria' in user_input_lower:
        return 'gender dysphoria'  
    elif 'sexual orientation' in user_input_lower:
        return 'sexual orientation struggles'  
    elif 'cultural pressure' in user_input_lower:
        return 'cultural pressure'  
    elif 'racial discrimination' in user_input_lower:
        return 'racial discrimination'  
    elif 'microaggressions' in user_input_lower:
        return 'microaggressions'  
    elif 'stereotype' in user_input_lower:
        return 'stereotyped'  
    elif 'language barrier' in user_input_lower:
        return 'language barrier'  
    elif 'homesick' in user_input_lower:
        return 'homesick'  
    elif 'culture shock' in user_input_lower:
        return 'culture shock'  
    elif 'immigration stress' in user_input_lower:
        return 'immigration stress'  
    elif 'legal issues' in user_input_lower:
        return 'legal issues'  
    elif 'court case' in user_input_lower:
        return 'court case stress'  
    elif 'debt' in user_input_lower:
        return 'debt stress'  
    elif 'bankruptcy' in user_input_lower:
        return 'bankruptcy'  
    elif 'housing issues' in user_input_lower:
        return 'housing issues'  
    elif 'homeless' in user_input_lower:
        return 'homeless'  
    elif 'relationship abuse' in user_input_lower:
        return 'relationship abuse'  
    elif 'gaslighting' in user_input_lower:
        return 'gaslighting'  
    elif 'manipulation' in user_input_lower:
        return 'manipulation'  
    elif 'toxic relationship' in user_input_lower:
        return 'toxic relationship'  
    elif 'friendship betrayal' in user_input_lower:
        return 'friendship betrayal'  
    elif 'sibling rivalry' in user_input_lower:
        return 'sibling rivalry'  
    elif 'parental expectations' in user_input_lower:
        return 'parental expectations'  
    elif 'caregiving stress' in user_input_lower:
        return 'caregiving stress'  
    elif 'elderly care' in user_input_lower:
        return 'elderly care stress'  
    elif 'childcare stress' in user_input_lower:
        return 'childcare stress'  
    elif 'academic pressure' in user_input_lower:
        return 'academic pressure'  
    elif 'exam stress' in user_input_lower:
        return 'exam stress'  
    elif 'failing grades' in user_input_lower:
        return 'failing grades'  
    elif 'school bullying' in user_input_lower:
        return 'school bullying'  
    elif 'peer pressure' in user_input_lower:
        return 'peer pressure'  
    elif 'social media pressure' in user_input_lower:
        return 'social media pressure'  
    elif 'online harassment' in user_input_lower:
        return 'online harassment'  
    elif 'cyberbullying' in user_input_lower:
        return 'cyberbullying'  
    elif 'technology overload' in user_input_lower:
        return 'technology overload'  
    elif 'screen time' in user_input_lower:
        return 'screen time stress'  
    elif 'pandemic stress' in user_input_lower:
        return 'pandemic stress'  
    elif 'quarantine' in user_input_lower:
        return 'quarantine stress'  
    elif 'health anxiety' in user_input_lower:
        return 'health anxiety'  
    elif 'chronic illness' in user_input_lower:
        return 'chronic illness'  
    elif 'disability' in user_input_lower:
        return 'disability challenges'  
    elif 'medical procedure' in user_input_lower:
        return 'medical procedure stress'  
    elif 'surgery recovery' in user_input_lower:
        return 'surgery recovery'  
    elif 'grieving pet' in user_input_lower:
        return 'grieving pet'  
    elif 'pet loss' in user_input_lower:
        return 'pet loss'  
    elif 'aging concerns' in user_input_lower:
        return 'aging concerns'  
    elif 'retirement stress' in user_input_lower:
        return 'retirement stress'  
    elif 'empty nest' in user_input_lower:
        return 'empty nest syndrome'  
    elif 'midlife crisis' in user_input_lower:
        return 'midlife crisis'  
    elif 'spirituality crisis' in user_input_lower:
        return 'spirituality crisis'  
    elif 'faith doubts' in user_input_lower:
        return 'faith doubts'  
    elif 'existential crisis' in user_input_lower:
        return 'existential crisis'  
    elif 'purpose in life' in user_input_lower:
        return 'lack of purpose'  
    elif 'climate anxiety' in user_input_lower:
        return 'climate anxiety'  
    elif 'environmental stress' in user_input_lower:
        return 'environmental stress'  
    elif 'natural disaster' in user_input_lower:
        return 'natural disaster stress'  
    elif 'war trauma' in user_input_lower:
        return 'war trauma'  
    elif 'political stress' in user_input_lower:
        return 'political stress'  
    elif 'disillusioned' in user_input_lower:
        return 'disillusioned'  # 211
    elif 'hopeless' in user_input_lower:
        return 'hopeless'  
    elif 'despair' in user_input_lower:
        return 'despair'  
    elif 'numb' in user_input_lower:
        return 'numb'  
    elif 'detached' in user_input_lower:
        return 'detached'  
    elif 'restless' in user_input_lower:
        return 'restless'  
    elif 'impatient' in user_input_lower:
        return 'impatient'  
    elif 'frantic' in user_input_lower:
        return 'frantic'  
    elif 'exhausted' in user_input_lower:
        return 'exhausted'  
    elif 'burnout' in user_input_lower:
        return 'burnout'  
    elif 'sleep issues' in user_input_lower:
        return 'sleep issues' 
    elif 'insomnia' in user_input_lower:
        return 'insomnia' 
    elif 'nightmares' in user_input_lower:
        return 'nightmares'  
    elif 'trauma' in user_input_lower:
        return 'trauma'  
    elif 'ptsd' in user_input_lower:
        return 'ptsd'  
    elif 'flashbacks' in user_input_lower:
        return 'flashbacks'  
    elif 'panic attack' in user_input_lower:
        return 'panic attack'  
    elif 'social anxiety' in user_input_lower:
        return 'social anxiety'
    elif 'public speaking' in user_input_lower:
        return 'public speaking fear' 
    elif 'performance anxiety' in user_input_lower:
        return 'performance anxiety' 
    elif 'imposter syndrome' in user_input_lower:
        return 'imposter syndrome'  
    elif 'perfectionism' in user_input_lower:
        return 'perfectionism'  
    elif 'procrastination' in user_input_lower:
        return 'procrastination'  
    elif 'time management' in user_input_lower:
        return 'time management stress'  
    elif 'decision fatigue' in user_input_lower:
        return 'decision fatigue'  
    elif 'trust issues' in user_input_lower:
        return 'trust issues'  
    elif 'attachment issues' in user_input_lower:
        return 'attachment issues'  
    elif 'abandonment' in user_input_lower:
        return 'abandonment fear'  
    elif 'jealousy' in user_input_lower:
        return 'jealousy'  
    elif 'insecurity' in user_input_lower:
        return 'insecurity' 
    elif 'self-doubt' in user_input_lower:
        return 'self-doubt'  
    elif 'identity crisis' in user_input_lower:
        return 'identity crisis'  
    elif 'generational trauma' in user_input_lower:
        return 'generational trauma'  
    elif 'family conflict' in user_input_lower:
        return 'family conflict'  
    elif 'divorce stress' in user_input_lower:
        return 'divorce stress'  
    elif 'custody battle' in user_input_lower:
        return 'custody battle stress'  
    elif 'financial insecurity' in user_input_lower:
        return 'financial insecurity'  
    elif 'job insecurity' in user_input_lower:
        return 'job insecurity'  
    elif 'unemployment' in user_input_lower:
        return 'unemployment stress'  
    elif 'career change' in user_input_lower:
        return 'career change stress' 
    elif 'hi' in user_input_lower and user_input_lower.index('hi') == 0:
        return 'greeting_hi'  
    elif 'hello' in user_input_lower and user_input_lower.index('hello') == 0:
        return 'greeting_hello'  
    elif 'how are you' in user_input_lower and user_input_lower.index('how are you') == 0:
        return 'greeting_how_are_you' 
    elif 'hey' in user_input_lower and user_input_lower.index('hey') == 0:
        return 'greeting_hey'  
    elif 'good morning' in user_input_lower and user_input_lower.index('good morning') == 0:
        return 'greeting_good_morning'  
    elif 'bye' in user_input_lower :
        return 'farewell_bye'  
    elif 'thanks' in user_input_lower :
        return 'farewell_thanks'  
    elif 'thank you' in user_input_lower :
        return 'farewell_thank_you'  
    elif 'oh' in user_input_lower :
        return 'farewell_oh'  
    elif 'see you' in user_input_lower :
        return 'farewell_see_you' 
    elif ('music' in user_input_lower or 'song' in user_input_lower) and ('stress' in user_input_lower or 'relax' in user_input_lower or 'calm' in user_input_lower):
        return "music_for_stress"
    elif "face" in user_input_lower and ("care" in user_input_lower or "skin" in user_input_lower or "product" in user_input_lower):
        return "face_care"
    elif "hair" in user_input_lower and ("growth" in user_input_lower or "grow" in user_input_lower or "loss" in user_input_lower or "care" in user_input_lower):
        return "hair_growth"
    elif "relieve" in user_input_lower and "stress" in user_input_lower:
        return "stress_relief_activity"
    elif "sleep" in user_input_lower and ("better" in user_input_lower or "improve" in user_input_lower):
        return "sleep_improvement"
    elif "drink" in user_input_lower and "water" in user_input_lower:
        return "hydration_reminder"
    elif "morning" in user_input_lower and "routine" in user_input_lower:
        return "morning_routine"
    elif "evening" in user_input_lower and "relax" in user_input_lower:
        return "evening_relaxation"
    elif "tired" in user_input_lower and ("do" in user_input_lower or "help" in user_input_lower):
        return "energy_boost"
    elif "focus" in user_input_lower and ("better" in user_input_lower or "improve" in user_input_lower):
        return "focus_improvement"
    elif "nail" in user_input_lower and "care" in user_input_lower:
        return "nail_care"
    elif "hand" in user_input_lower and "dry" in user_input_lower:
        return "dry_hands"
    elif "feet" in user_input_lower and "care" in user_input_lower:
        return "foot_care"
    elif "back" in user_input_lower and ("hurt" in user_input_lower or "pain" in user_input_lower):
        return "back_pain_relief"
    elif "eye" in user_input_lower and ("tired" in user_input_lower or "strain" in user_input_lower):
        return "eye_strain"
    elif "snack" in user_input_lower :
        return "healthy_snack"
    elif "mood" in user_input_lower and ("improve" in user_input_lower or "better" in user_input_lower):
        return "mood_booster"
    elif "eating" in user_input_lower :
        return "stress_eating"
    elif "sunscreen" in user_input_lower or ("sun" in user_input_lower and "protect" in user_input_lower):
        return "sun_protection"
    elif "posture" in user_input_lower and ("improve" in user_input_lower or "better" in user_input_lower):
        return "posture_improvement"
    elif "lip" in user_input_lower and ("chapped" in user_input_lower or "dry" in user_input_lower):
        return "lip_care"
    elif "scalp" in user_input_lower and ("care" in user_input_lower or "healthy" in user_input_lower):
        return "scalp_health"
    elif "energy drink" in user_input_lower:
        return "energy_drink_alternative"
    elif "mindful" in user_input_lower or "mindfulness" in user_input_lower:
        return "mindfulness_practice"
    elif "teeth" in user_input_lower and "care" in user_input_lower:
        return "teeth_care"
    elif "hand" in user_input_lower and "cold" in user_input_lower:
        return "cold_hands"
    elif "stomach" in user_input_lower and ("upset" in user_input_lower or "hurt" in user_input_lower):
        return "stomach_upset"
    elif "neck" in user_input_lower and ("hurt" in user_input_lower or "pain" in user_input_lower):
        return "neck_pain"
    elif "eye" in user_input_lower and "dry" in user_input_lower:
        return "dry_eyes"
    elif "breakfast" in user_input_lower:
        return "healthy_breakfast"
    elif "journal" in user_input_lower and "stress" in user_input_lower:
        return "stress_journaling"
    elif ("phone" in user_input_lower or "screen" in user_input_lower) and ("too long" in user_input_lower or "break" in user_input_lower):
        return "screen_time_break"
    elif "hand" in user_input_lower and "tense" in user_input_lower:
        return "hand_relaxation"
    elif "leg" in user_input_lower and "cramp" in user_input_lower:
        return "leg_cramps"
    elif "morning" in user_input_lower and "energy" in user_input_lower:
        return "morning_energy"
    elif "wind down" in user_input_lower or ("night" in user_input_lower and "relax" in user_input_lower):
        return "evening_wind_down"
    elif "skin" in user_input_lower and "dry" in user_input_lower:
        return "dry_skin"
    elif "headache" in user_input_lower or ("head" in user_input_lower and "hurt" in user_input_lower):
        return "headache_relief"
    elif "feet" in user_input_lower and ("smell" in user_input_lower or "odor" in user_input_lower):
        return "foot_odor"
    elif "mood" in user_input_lower and "track" in user_input_lower:
        return "mood_tracking"
    return 'bothered'

def generate_response_rule_based(user_input):
    is_offensive, message = filter_offensive_input(user_input)
    if is_offensive:
        return message
    
    emotion = extract_emotion(user_input)
    user_input_lower = user_input.lower()
    # Emotional Scenarios
    if emotion == 'angry':
        fallback_response = "I can see that you're feeling angry, and it’s okay to feel this way sometimes. Try taking a few deep breaths or stepping away for a moment to cool down. Would you like to talk about what’s making you angry?"
    elif emotion == 'frustrated':
        fallback_response = "I understand that you’re feeling frustrated, which can be really tough. Maybe take a short break and write down what’s bothering you to help clear your mind. Can you share more about what’s going on?"
    elif emotion == 'overwhelmed':
        fallback_response = "It sounds like you’re feeling overwhelmed, and I’m here to help you through this. Try breaking your tasks into smaller, manageable steps and tackle them one at a time. Would you like to discuss what’s overwhelming you?"
    elif emotion == 'stressed':
        fallback_response = "I’m sorry to hear that you’re feeling stressed—it can be really challenging. Consider taking a few minutes to practice deep breathing or go for a short walk to help ease your mind. Would you like to talk more about what’s stressing you out?"
    elif emotion == 'anxious':
        fallback_response = "I can tell you’re feeling anxious, and I’m here for you. Try focusing on something grounding, like feeling your feet on the floor or counting your breaths. Would you like to share what’s making you anxious?"
    elif emotion == 'scared':
        fallback_response = "I understand that you’re feeling scared, which can be really overwhelming. It might help to take a few deep breaths and focus on something calming, like listening to soothing music or talking to someone you trust. Would you like to share more about what’s making you feel this way?"
    elif emotion == 'terrified':
        fallback_response = "I’m so sorry you’re feeling terrified—that must be incredibly difficult. Let’s try to ground yourself by focusing on your breathing or holding onto something comforting, like a soft blanket. Can you tell me more so I can support you?"
    elif emotion == 'nervous':
        fallback_response = "I can see that you’re feeling nervous, and that’s completely okay. Try visualizing a positive outcome or doing a quick relaxation exercise, like stretching your hands. Would you like to talk about what’s making you nervous?"
    elif emotion == 'worried':
        fallback_response = "I’m sorry to hear that you’re feeling worried—it can weigh heavily on you. Writing down your worries and addressing them one by one might help you feel more in control. Can we talk about what’s worrying you?"
    elif emotion == 'panicked':
        fallback_response = "I’m here for you—it sounds like you might be feeling panicked, which can be really intense. Try breathing in for 4 seconds, holding for 4, and exhaling for 4 to help calm your body. Would you like to talk through what’s happening?"
    elif emotion == 'sad':
        fallback_response = "I’m really sorry to hear that you’re feeling sad—I’m here to support you. Maybe doing something comforting, like listening to your favorite music or talking to a loved one, can help lift your spirits. Would you like to share more about how you’re feeling?"
    elif emotion == 'depressed':
        fallback_response = "I’m so sorry you’re feeling depressed—it can be really hard, and I’m here for you. Try to do one small thing that brings you comfort, like taking a warm shower or reaching out to someone you trust. Would you like to talk more about what’s going on?"
    elif emotion == 'hopeless':
        fallback_response = "I can hear how hopeless you’re feeling right now, and I’m here to help you through this. Even small steps, like getting some fresh air or writing down one thing you’re grateful for, can make a difference. Can we talk more about what’s making you feel this way?"
    elif emotion == 'lonely':
        fallback_response = "I’m sorry you’re feeling lonely—it can be really tough to feel this way. Maybe try reaching out to a friend or family member, even just to say hi, or engage in an activity you enjoy. Would you like to talk more about how you’re feeling?"
    elif emotion == 'isolated':
        fallback_response = "I can tell you’re feeling isolated, and I’m here to support you. Connecting with someone, even through a quick message or joining an online community, might help you feel less alone. Would you like to share more about what’s going on?"
    elif emotion == 'grieving':
        fallback_response = "I’m so sorry for your loss—grieving can be incredibly hard, and I’m here for you. It might help to honor your loved one’s memory by sharing stories or creating a small ritual, like lighting a candle. Would you like to talk more about them?"
    elif emotion == 'heartbroken':
        fallback_response = "I’m really sorry you’re feeling heartbroken—it’s such a painful experience. Try to be gentle with yourself by doing something soothing, like watching a comforting movie or writing down your feelings. Would you like to share more about what happened?"
    elif emotion == 'guilty':
        fallback_response = "I can see that you’re feeling guilty, and I’m here to help you through this. Reflecting on what you can learn from the situation and making amends, if possible, might help ease your feelings. Would you like to talk more about what’s making you feel this way?"
    elif emotion == 'ashamed':
        fallback_response = "I’m sorry you’re feeling ashamed—it can be really tough to carry that feeling. Remember that everyone makes mistakes, and it might help to talk to someone you trust or write down your thoughts to process them. Can you share more about what’s going on?"
    elif emotion == 'embarrassed':
        fallback_response = "I can tell you’re feeling embarrassed, and I’m here for you—it’s okay to feel this way sometimes. Try to remind yourself that everyone has moments like this, and maybe do something kind for yourself, like taking a relaxing break. Would you like to talk about what happened?"
    elif emotion == 'jealous':
        fallback_response = "I understand that you’re feeling jealous, and it’s okay to have these feelings. Try focusing on what you’re grateful for in your own life, or talk to someone about how you’re feeling to gain perspective. Would you like to share more about what’s making you feel this way?"
    elif emotion == 'insecure':
        fallback_response = "I’m sorry you’re feeling insecure—I’m here to support you. It might help to write down three things you appreciate about yourself, or talk to someone who makes you feel valued. Would you like to share more about what’s making you feel insecure?"
    elif emotion == 'confused':
        fallback_response = "I can see that you’re feeling confused, and I’m here to help you sort things out. Maybe try writing down your thoughts or breaking the situation into smaller parts to make it clearer. Would you like to talk more about what’s confusing you?"
    elif emotion == 'lost':
        fallback_response = "I’m sorry you’re feeling lost—it can be really disorienting. Try focusing on one small goal or routine, like going for a walk, to help you feel more grounded. Would you like to share more about what’s making you feel this way?"
    elif emotion == 'empty':
        fallback_response = "I can tell you’re feeling empty, and I’m here for you—it’s a tough feeling to experience. Maybe try doing something small that brings you comfort, like listening to music or spending time with a pet. Would you like to talk more about how you’re feeling?"
    elif emotion == 'numb':
        fallback_response = "I’m sorry you’re feeling numb—it can be really hard to feel disconnected like this. Try engaging your senses, like holding something warm or smelling a calming scent, to help you reconnect. Would you like to share more about what’s going on?"
    elif emotion == 'exhausted':
        fallback_response = "I can tell you’re feeling exhausted, and I’m here for you—it’s tough to feel so drained. Make sure to rest and hydrate, and maybe set aside some time for a relaxing activity, like a short nap or a warm bath. Would you like to talk more about what’s been tiring you out?"
    elif emotion == 'burned out':
        fallback_response = "I’m sorry to hear you’re feeling burned out—it can be really overwhelming. Try to take a break and do something that recharges you, like spending time in nature or doing a hobby you enjoy. Would you like to talk more about what’s been going on?"
    elif emotion == 'unmotivated':
        fallback_response = "I can see that you’re feeling unmotivated, and I’m here to help you through this. Start with a small, achievable task to build momentum, like organizing your desk or taking a short walk. Would you like to share more about what’s making you feel this way?"
    elif emotion == 'restless':
        fallback_response = "I understand you’re feeling restless—it can be hard to feel so unsettled. Try channeling that energy into something physical, like stretching or a quick workout, to help calm your mind. Would you like to talk more about what’s going on?"
    # Mental Health Conditions and Symptoms
    elif emotion == 'anxiety':
        fallback_response = "I’m sorry to hear you’re struggling with anxiety—I’m here to support you. Try grounding yourself by focusing on your breathing or naming five things you can see around you. Would you like to talk more about what’s been triggering your anxiety?"
    elif emotion == 'depression':
        fallback_response = "I’m really sorry you’re dealing with depression—it can be so challenging, and I’m here for you. Even a small step, like getting out of bed or talking to someone you trust, can make a difference. Would you like to share more about how you’re feeling?"
    elif emotion == 'ptsd':
        fallback_response = "I’m so sorry you’re experiencing PTSD—it can be really tough, and I’m here to help. Try focusing on a grounding technique, like feeling a textured object, to bring you back to the present. Would you like to talk more about what’s been coming up for you?"
    elif emotion == 'ocd':
        fallback_response = "I can tell you’re struggling with OCD, and I’m here to support you—it can be really overwhelming. Try to gently redirect your focus to a calming activity, like drawing or listening to music, to help manage your thoughts. Would you like to share more about what you’re experiencing?"
    elif emotion == 'bipolar':
        fallback_response = "I’m sorry to hear you’re dealing with bipolar disorder or mood swings—I’m here for you. Keeping a mood journal or sticking to a consistent routine might help stabilize your emotions. Would you like to talk more about how you’re feeling right now?"
    elif emotion == 'schizophrenia':
        fallback_response = "I’m here for you—it sounds like you’re dealing with schizophrenia, which can be really challenging. It might help to reach out to a trusted mental health professional or stick to a calming routine to feel more grounded. Would you like to share more about what’s going on?"
    elif emotion == 'panic attacks':
        fallback_response = "I’m so sorry you’re experiencing panic attacks—they can be really intense, and I’m here for you. Try breathing slowly—inhale for 4 seconds, hold for 4, and exhale for 4—to help your body calm down. Would you like to talk more about what’s been triggering them?"
    elif emotion == 'social anxiety':
        fallback_response = "I can tell you’re dealing with social anxiety, and I’m here to support you—it can be really tough. Practice a small social interaction, like smiling at someone, to build confidence gradually. Would you like to talk more about what’s been challenging for you?"
    elif emotion == 'having trouble sleeping':
        fallback_response = "I understand that you’re having trouble sleeping, which can be really tough. Try creating a calming bedtime routine, like avoiding screens and drinking a warm cup of chamomile tea. Would you like to talk more about what’s keeping you up?"
    elif emotion == 'loss of appetite':
        fallback_response = "I’m sorry to hear you’re experiencing a loss of appetite—it can be concerning, and I’m here for you. Try eating small, nutritious snacks, like fruit or nuts, and stay hydrated to support your body. Would you like to talk more about what’s been going on?"
    elif emotion == 'crying spells':
        fallback_response = "I can tell you’re having cryingSpells, and I’m here to support you—it’s okay to let your emotions out. Maybe find a quiet space to process your feelings, or talk to someone you trust for comfort. Would you like to share more about what’s been happening?"
    elif emotion == 'trouble concentrating':
        fallback_response = "I’m sorry you’re having trouble concentrating—it can be really frustrating. Try breaking your tasks into smaller chunks and setting a timer for focused work, like 10 minutes at a time. Would you like to talk more about what’s been distracting you?"
    elif emotion == 'self-harm':
        fallback_response = "I’m really concerned to hear you’re struggling with self-harm, and I’m here for you—you’re not alone. Try a safer coping mechanism, like holding an ice cube or drawing on your skin with a marker, and please consider reaching out to a trusted person or professional. Can we talk more about how you’re feeling?"
    elif emotion == 'suicidal thoughts':
        fallback_response = "I’m so sorry you’re having suicidal thoughts—I’m here for you, and you’re not alone in this. Please reach out to a crisis hotline, like the National Suicide Prevention Lifeline at 1-800-273-8255, or someone you trust right now. Can we talk more about what’s been going on?"
    elif emotion == 'mental breakdown':
        fallback_response = "I’m really sorry you’re experiencing a mental breakdown—it can feel overwhelming, and I’m here for you. Try to find a quiet space and focus on slow breathing to help calm your mind, and consider reaching out to a professional for support. Would you like to share more about what’s happening?"
    elif emotion == 'overthinking':
        fallback_response = "I can tell you’re overthinking, and I’m here to help you through this—it can be exhausting. Try writing down your thoughts to get them out of your head, or distract yourself with a calming activity like listening to music. Would you like to talk more about what’s on your mind?"
    elif emotion == 'intrusive thoughts':
        fallback_response = "I’m sorry you’re dealing with intrusive thoughts—they can be really distressing, and I’m here for you. Try acknowledging the thought without judgment and then redirecting your focus to something grounding, like counting your breaths. Would you like to talk more about what’s been coming up?"
    elif emotion == 'dissociation':
        fallback_response = "I can tell you’re experiencing dissociation, and I’m here to support you—it can feel really unsettling. Try grounding yourself by touching something textured or naming five things you can see around you to bring you back to the present. Would you like to share more about how you’re feeling?"
    elif emotion == 'hyperventilating':
        fallback_response = "I’m sorry you’re hyperventilating—it can feel really scary, and I’m here for you. Try breathing into a paper bag or slowly inhaling through your nose for 4 seconds and exhaling for 6 to help regulate your breathing. Would you like to talk more about what’s happening?"
    elif emotion == 'flashbacks':
        fallback_response = "I’m so sorry you’re experiencing flashbacks—they can be really intense, and I’m here for you. Try grounding yourself by focusing on your senses, like smelling something calming or touching a soft object, to bring you back to the present. Would you like to talk more about what’s coming up for you?"
    # Relationship and Social Scenarios
    elif emotion == 'anxious about friendship':
        if 'value' in user_input_lower or 'not valued' in user_input_lower:
            fallback_response = "I’m sorry you feel like your friends don’t value you—that must be really hurtful, and I’m here for you. Consider talking to them about how their actions make you feel, or focus on spending time with people who make you feel appreciated. Would you like to explore ways to address this?"
        elif 'cheating' in user_input_lower:
            fallback_response = "I can tell you’re feeling hurt about possible cheating in your friendship, and I’m here for you—it’s a tough situation. Try having an honest conversation with your friend to understand what happened, and give yourself time to process your feelings. Would you like to talk more about what’s going on?"
        else:
            fallback_response = "I understand that you're feeling anxious about friendship, and I’m here to help—it can be hard when relationships feel uncertain. Maybe try having an open conversation with your friend about how you’re feeling to strengthen your connection. Would you like to share more about what’s going on?"
    elif emotion == 'friends don’t value me':
        fallback_response = "I’m sorry you feel like your friends don’t value you—that must be really hurtful, and I’m here for you. Consider talking to them about how their actions make you feel, or focus on spending time with people who make you feel appreciated. Would you like to explore ways to address this?"
    elif emotion == 'cheating':
        fallback_response = "I can tell you’re feeling hurt about possible cheating in your relationship, and I’m here for you—it’s a tough situation. Try having an honest conversation with the person to understand what happened, and give yourself time to process your feelings. Would you like to talk more about what’s going on?"
    elif emotion == 'breakup':
        fallback_response = "I’m so sorry you’re going through a breakup—it can be really painful, and I’m here for you. Be gentle with yourself by doing things that bring you comfort, like spending time with loved ones or journaling your feelings. Would you like to share more about what happened?"
    elif emotion == 'divorce':
        fallback_response = "I’m really sorry you’re dealing with a divorce or separation—it’s a challenging time, and I’m here to support you. Try focusing on self-care, like establishing a new routine or talking to a trusted friend, to help you adjust. Would you like to talk more about how you’re feeling?"
    elif emotion == 'lonely in relationship':
        fallback_response = "I’m sorry you’re feeling lonely in your relationship—that must be really hard, and I’m here for you. Try communicating your feelings with your partner to see if you can reconnect, or spend time doing things that make you feel fulfilled on your own. Would you like to talk more about what’s been going on?"
    elif emotion == 'family issues':
        fallback_response = "I can tell you’re dealing with family issues, and I’m here to support you—it can be really stressful. Maybe try setting boundaries or having a calm conversation with a family member to address the problem. Would you like to share more about what’s happening?"
    elif emotion == 'parenting stress':
        fallback_response = "I’m sorry you’re feeling stressed about parenting—it can be a lot to handle, and I’m here for you. Try taking a short break to recharge, and maybe involve your kids in a fun activity to connect with them. Would you like to talk more about what’s been challenging?"
    elif emotion == 'workplace conflict':
        fallback_response = "I can tell you’re dealing with workplace conflict, and I’m here to help—it can be really tough. Try addressing the issue calmly with the person involved, or speak to a supervisor if needed to find a resolution. Would you like to share more about what’s going on at work?"
    elif emotion == 'bullying':
        fallback_response = "I’m so sorry you’re experiencing bullying—it’s not okay, and I’m here for you. Consider talking to a trusted person, like a teacher or supervisor, to help address the situation, and focus on activities that make you feel safe and supported. Would you like to talk more about what’s been happening?"
    # Work and Lifestyle Scenarios
    elif emotion == 'overwhelmed by workload':
        fallback_response = "I understand that you're feeling overwhelmed by workload, which can make maintaining a work-life balance really challenging. Try setting clear boundaries, like scheduling specific times for work and relaxation, and take short breaks to recharge during the day. Would you like to talk more about your workload?"
    elif emotion == 'job stress':
        fallback_response = "I’m sorry you’re dealing with job stress—it can be really taxing, and I’m here for you. Try organizing your tasks with a to-do list and taking short breaks to breathe deeply or stretch. Would you like to talk more about what’s been stressful at work?"
    elif emotion == 'unemployment':
        fallback_response = "I’m really sorry to hear you’re dealing with unemployment—it’s a tough situation, and I’m here for you. Try focusing on small steps, like updating your resume or reaching out to your network, to help you feel more in control. Would you like to talk more about what happened?"
    elif emotion == 'financial stress':
            fallback_response = "I can tell you’re feeling stressed about finances, and I’m here to support you—it can be really overwhelming. Try creating a simple budget or reaching out to a financial advisor for guidance, and take it one step at a time. Would you like to talk more about what’s going on?"
    elif emotion == 'seeking daily routine':
        fallback_response = "Establishing a daily health routine can really help you relax and manage stress. Start your day with 10 minutes of stretching or meditation, take short breaks to breathe deeply or walk outside, and end your day with a calming activity like reading or a warm bath. Would you like to explore more relaxation techniques?"
    elif emotion == 'procrastination':
        fallback_response = "I can see you’re struggling with procrastination, and I’m here to help—it can be tough to get started. Try setting a small, specific goal and using a timer for just 5 minutes to begin, which can help build momentum. Would you like to talk more about what you’re procrastinating on?"
    elif emotion == 'time management':
        fallback_response = "I’m here for you—it sounds like you’re struggling with time management, which can feel overwhelming. Try prioritizing your tasks with a daily planner and setting specific time blocks for each activity to stay organized. Would you like to talk more about your schedule?"
    elif emotion == 'lack of energy':
        fallback_response = "I’m sorry you’re feeling a lack of energy—it can be really draining, and I’m here for you. Make sure you’re staying hydrated and eating nutritious meals, and maybe try a short walk to boost your energy. Would you like to talk more about what’s been going on?"
    elif emotion == 'exercise motivation':
        fallback_response = "I can tell you’re looking for motivation to exercise, and I’m here to help—it’s great that you’re thinking about it. Start with something small, like a 10-minute walk, and choose an activity you enjoy to make it more fun. Would you like to talk more about what kind of exercise you’d like to try?"
    elif emotion == 'healthy eating':
        fallback_response = "I’m here to support you—it’s great that you’re thinking about healthy eating. Try incorporating more whole foods, like fruits and vegetables, into your meals, and plan your meals ahead to make it easier. Would you like to talk more about your eating habits?"
    # Physical Health and Self-Care
    elif emotion == 'in pain':
        fallback_response = "I’m really sorry to hear that you’re feeling in pain—it must be tough, and I’m here for you. It might help to rest and sip some warm water, but I’d also recommend checking with a doctor if the pain persists. Would you like to share more about how you’re feeling?"
    elif emotion == 'headache':
        fallback_response = "I’m sorry you’re dealing with a headache—that can be really uncomfortable, and I’m here for you. Try resting in a dark, quiet room and drinking some water, as dehydration can sometimes be a cause. Would you like to talk more about what might have triggered it?"
    elif emotion == 'stomach pain':
        fallback_response = "I’m really sorry you’re experiencing stomach pain—I’m here to support you. Try resting and sipping warm water or herbal tea, but if it continues, it’s best to consult a doctor. Would you like to talk more about how you’re feeling?"
    elif emotion == 'chronic pain':
        fallback_response = "I’m so sorry you’re dealing with chronic pain—it can be really challenging, and I’m here for you. Consider gentle stretching or a warm compress to help ease the discomfort, and talking to a doctor about pain management might help. Would you like to share more about what you’ve been experiencing?"
    elif emotion == 'self-care':
        fallback_response = "I’m glad you’re thinking about self-care—it’s so important, and I’m here to help. Try setting aside time each day for something you enjoy, like reading, taking a bath, or going for a walk. Would you like to talk more about what self-care looks like for you?"
    elif emotion == 'mindfulness':
        fallback_response = "I’m here for you—it’s great that you’re interested in mindfulness. Try a simple practice, like focusing on your breath for 5 minutes or noticing the sensations around you, to help you feel more present. Would you like to explore more mindfulness techniques?"
    elif emotion == 'meditation':
        fallback_response = "I’m glad you’re considering meditation—it can be really calming, and I’m here to help. Start with a short session, like 5 minutes of focusing on your breath, and find a quiet space where you feel comfortable. Would you like to talk more about how to get started?"
    elif emotion == 'deep breathing':
        fallback_response = "I’m here to support you—deep breathing is a great way to relax. Try inhaling for 4 seconds, holding for 4, and exhaling for 4, and repeat a few times to help calm your mind. Would you like to practice together or talk more about it?"
    elif emotion == 'journaling':
        fallback_response = "I’m glad you’re thinking about journaling—it can be a great way to process your feelings, and I’m here for you. Try writing down three things you’re grateful for or how you’re feeling today to get started. Would you like to talk more about what you’d like to journal about?"
    # Seeking Professional Help
    elif emotion == 'considering therapy':
        fallback_response = "It’s great that you’re considering therapy to help with what you’re going through—I’m here for you. A good starting point could be cognitive-behavioral therapy (CBT), which helps manage anxiety and negative thoughts, or mindfulness-based therapy for relaxation. Would you like help finding a therapist to explore these options?"
    elif emotion == 'counseling':
        fallback_response = "I’m glad you’re thinking about counseling—it can be really helpful, and I’m here to support you. You might look for a counselor who specializes in your concerns, like anxiety or grief, to get the best support. Would you like to talk more about what you’re looking for in counseling?"
    elif emotion == 'support groups':
        fallback_response = "I’m here for you—it’s great that you’re considering support groups, as they can provide a sense of community. Look for local or online groups that focus on what you’re going through, like anxiety or grief support. Would you like help finding a group that fits your needs?"
    elif emotion == 'crisis hotline':
        fallback_response = "I’m here for you, and I’m glad you’re seeking urgent help—you’re not alone. Please call a crisis hotline, like the National Suicide Prevention Lifeline at 1-800-273-8255, or reach out to someone you trust right away. Would you like to talk more while you get support?"
    elif emotion == 'helpline':
        fallback_response = "I’m here to support you—it’s great that you’re considering a helpline for help. You can call a helpline like the National Alliance on Mental Illness (NAMI) at 1-800-950-6264 for support and resources. Would you like to talk more about what’s going on?"
    elif emotion == 'emergency contact':
        fallback_response = "I’m really concerned for your safety, and I’m here for you—if you’re in immediate danger, please call emergency services like 911 right away. You can also reach out to a trusted person to help you through this. Can we talk more about what’s happening so I can support you?"
    elif emotion == 'psychiatrist':
        fallback_response = "I’m glad you’re considering seeing a psychiatrist—it can be a helpful step, and I’m here for you. They can assist with medication management or therapy recommendations, so look for one who specializes in your concerns. Would you like help finding a psychiatrist?"
    elif emotion == 'psychologist':
        fallback_response = "I’m here to support you—it’s great that you’re thinking about seeing a psychologist. They can help with therapy techniques like CBT or talk therapy, so look for one who fits your needs. Would you like to talk more about what you’re looking for?"
    elif emotion == 'diagnosis':
        fallback_response = "I can tell you’re thinking about a diagnosis, and I’m here for you—it’s a brave step to seek clarity. A mental health professional, like a psychologist or psychiatrist, can help evaluate your symptoms and provide guidance. Would you like to talk more about what you’ve been experiencing?"
    elif emotion == 'prescription':
        fallback_response = "I’m here for you—it sounds like you’re considering medication, which can be helpful for many people. A psychiatrist can help determine if a prescription is right for you and monitor how it works. Would you like to talk more about your concerns or experiences with medication?"
    # Other Mental Health Support Scenarios
    elif emotion == 'relapse':
        fallback_response = "I’m so sorry you’re experiencing a relapse—it’s a challenging moment, and I’m here for you. Try reaching out to your support system or a professional to help you get back on track, and be gentle with yourself. Would you like to talk more about what’s been happening?"
    elif emotion == 'therapy plan':
        fallback_response = "I’m glad you’re thinking about a therapy plan—it’s a great step, and I’m here to support you. You might work with your therapist to set goals, like managing anxiety or improving sleep, and track your progress together. Would you like to talk more about what you’d like to achieve?"
    elif emotion == 'side effects':
        fallback_response = "I’m sorry you’re dealing with side effects from medication—it can be tough, and I’m here for you. Consider talking to your doctor about adjusting your dosage or exploring other options to manage the side effects. Would you like to share more about what you’re experiencing?"
    elif emotion == 'treatment':
        fallback_response = "I’m here for you—it’s great that you’re thinking about treatment options for your mental health. Treatment might include therapy, medication, or lifestyle changes, so talking to a professional can help you find what works best. Would you like to talk more about what you’re considering?"
    elif emotion == 'mental health professional':
        fallback_response = "I’m glad you’re considering a mental health professional—it’s a positive step, and I’m here for you. Look for someone who specializes in your concerns, like a therapist for anxiety or a psychiatrist for medication management. Would you like help finding the right professional?"
    elif emotion == 'coping strategies':
        fallback_response = "I’m here to help—it’s great that you’re looking for coping strategies to manage how you’re feeling. Some strategies include deep breathing, journaling, or talking to a trusted friend—let’s find what works for you. Would you like to explore more ideas?"
    elif emotion == 'stress management':
        fallback_response = "I’m here for you—managing stress is so important, and I’m glad you’re thinking about it. Try techniques like deep breathing, setting realistic goals, or taking short breaks to help reduce stress. Would you like to talk more about what’s been stressful for you?"
    elif emotion == 'emotional regulation':
        fallback_response = "I can tell you’re interested in emotional regulation, and I’m here to support you—it’s a great skill to develop. Try naming your emotions and using grounding techniques, like focusing on your breath, to help manage them. Would you like to explore more techniques?"
    elif emotion == 'self-esteem':
        fallback_response = "I’m sorry you’re struggling with self-esteem—I’m here for you, and you’re enough just as you are. Try writing down three things you like about yourself or doing something you’re good at to boost your confidence. Would you like to talk more about how you’re feeling?"
    elif emotion == 'motivation':
        fallback_response = "I can tell you’re looking for motivation, and I’m here to help—it’s okay to feel this way sometimes. Start with a small goal, like taking a short walk, and celebrate your progress to build momentum. Would you like to talk more about what you’d like to achieve?"
    # Additional Scenarios
    elif emotion == 'fear of failure':
        fallback_response = "I’m sorry you’re dealing with a fear of failure—it can be really paralyzing, and I’m here for you. Try reframing failure as a learning opportunity and start with small, manageable steps toward your goal. Would you like to talk more about what you’re afraid of?"
    elif emotion == 'perfectionism':
        fallback_response = "I can tell you’re struggling with perfectionism, and I’m here to support you—it can be exhausting. Try setting realistic goals and reminding yourself that progress, not perfection, is what matters. Would you like to talk more about what’s been challenging?"
    elif emotion == 'grief anniversary':
        fallback_response = "I’m so sorry you’re feeling the weight of a grief anniversary—it can bring up a lot of emotions, and I’m here for you. Maybe honor the day by doing something meaningful, like visiting a special place or writing a letter to your loved one. Would you like to share more about how you’re feeling?"
    elif emotion == 'seasonal depression':
        fallback_response = "I’m sorry you’re experiencing seasonal depression—it can be really tough, especially during certain times of the year. Try getting as much natural light as possible, or consider talking to a professional about light therapy. Would you like to talk more about how you’re feeling?"
    elif emotion == 'identity crisis':
        fallback_response = "I can tell you’re going through an identity crisis, and I’m here for you—it can be really unsettling to feel this way. Try exploring your values and interests through journaling or talking to someone you trust to help you reconnect with yourself. Would you like to share more about what’s been on your mind?"
    elif emotion == 'trust issues':
        fallback_response = "I’m sorry you’re dealing with trust issues—it can make relationships really challenging, and I’m here for you. Start by setting small boundaries and communicating openly with someone you feel safe with to rebuild trust gradually. Would you like to talk more about what’s been going on?"
    elif emotion == 'abandonment fears':
        fallback_response = "I can tell you’re struggling with fears of abandonment, and I’m here to support you—it’s a tough feeling to navigate. Try focusing on building a strong support system and reminding yourself of your worth, even when others aren’t around. Would you like to share more about what’s triggering these fears?"
    elif emotion == 'codependency':
        fallback_response = "I’m here for you—it sounds like you’re dealing with codependency, which can be really challenging in relationships. Try focusing on your own needs and hobbies to build independence, and consider talking to a therapist about setting healthy boundaries. Would you like to talk more about your situation?"
    elif emotion == 'anger management':
        fallback_response = "I’m glad you’re thinking about anger management—it’s a brave step, and I’m here for you. Try counting to 10 before reacting or stepping away to cool down when you feel angry to help manage your emotions. Would you like to explore more techniques?"
    elif emotion == 'self-compassion':
        fallback_response = "I’m here to support you—it’s wonderful that you’re thinking about self-compassion. Try speaking to yourself as you would a dear friend, and give yourself permission to rest when you need it. Would you like to talk more about how you can be kinder to yourself?"
    # Scenario 
    # New Scenarios 
    elif emotion == 'disappointed':
        fallback_response = "I’m sorry you’re feeling disappointed—it can be really tough when things don’t go as hoped. Reflect on what you can learn from this experience, and maybe try something small that brings you joy to lift your spirits. Would you like to talk more about what happened?"  
    elif emotion == 'regretful':
        fallback_response = "I can tell you’re feeling regretful, and I’m here for you—regret can weigh heavily. Try to focus on what you can do now to move forward, like making amends or learning from the experience. Would you like to share more about what you’re regretting?"  
    elif emotion == 'shocked':
        fallback_response = "I’m sorry you’re feeling shocked—that can be really unsettling, and I’m here to support you. Take a moment to breathe deeply and process what happened, and maybe talk to someone you trust to help you make sense of it. Would you like to tell me more?"  
    elif emotion == 'disgusted':
        fallback_response = "I can see you’re feeling disgusted, and I’m here for you—it’s tough to feel this way. Try stepping away from the situation and doing something cleansing, like taking a walk or washing your hands, to help reset. Would you like to share what’s making you feel this way?"  
    elif emotion == 'irritated':
        fallback_response = "I’m sorry you’re feeling irritated—I’m here to help you through this. Maybe take a short break and do something calming, like listening to music, to help ease your frustration. Would you like to talk more about what’s irritating you?"  
    elif emotion == 'bored':
        fallback_response = "I can tell you’re feeling bored, and I’m here for you—it can be hard to feel unengaged. Try picking up a new hobby or activity, like drawing or exploring a new podcast, to spark some interest. Would you like to brainstorm some ideas together?"  
    elif emotion == 'apathetic':
        fallback_response = "I’m sorry you’re feeling apathetic—it can be tough to feel so disconnected, and I’m here for you. Start with a small, meaningful task, like making your bed, to help you feel a sense of accomplishment. Would you like to talk more about what’s going on?"  
    elif emotion == 'helpless':
        fallback_response = "I can tell you’re feeling helpless, and I’m here to support you—it’s hard to feel this way. Try focusing on one small thing you can control right now, like organizing your space, to regain a sense of agency. Would you like to share more about what’s making you feel this way?"  
    elif emotion == 'powerless':
        fallback_response = "I’m sorry you’re feeling powerless—it can be really overwhelming, and I’m here for you. Let’s identify one small action you can take to feel more in control, like writing down your thoughts or making a plan. Would you like to talk more about what’s going on?"  
    elif emotion == 'trapped':
        fallback_response = "I can see you’re feeling trapped, and I’m here to help—it’s tough to feel stuck like this. Try imagining a way out by brainstorming small steps to change your situation, or talk to someone who can offer a fresh perspective. Would you like to share more about what’s making you feel trapped?"  
    elif emotion == 'resentful':
        fallback_response = "I’m sorry you’re feeling resentful—it can be heavy to carry those feelings, and I’m here for you. Writing down what’s causing your resentment and talking it out with someone might help you process it. Would you like to share more about what’s going on?"  
    elif emotion == 'bitter':
        fallback_response = "I can tell you’re feeling bitter, and I’m here to support you—it’s hard to feel this way. Try focusing on something positive in your life, like a hobby or a loved one, to help shift your perspective. Would you like to talk more about what’s making you feel bitter?"  
    elif emotion == 'envious':
        fallback_response = "I’m sorry you’re feeling envious—I’m here for you, and it’s okay to feel this way sometimes. Try celebrating your own achievements and focusing on what makes you unique to help ease those feelings. Would you like to share more about what’s triggering your envy?"  
    elif emotion == 'doubtful':
        fallback_response = "I can see you’re feeling doubtful, and I’m here to help—it’s tough to feel uncertain. Try listing the facts you know to ground yourself, or talk to someone who can offer reassurance. Would you like to talk more about what’s making you doubt?"  
    elif emotion == 'skeptical':
        fallback_response = "I’m sorry you’re feeling skeptical—it can be hard to trust, and I’m here for you. Maybe take a step back and look for evidence to help you decide what to believe, or discuss your thoughts with someone you trust. Would you like to share more about what’s making you skeptical?"  
    elif emotion == 'paranoid':
        fallback_response = "I can tell you’re feeling paranoid, and I’m here to support you—it can be really unsettling. Try grounding yourself by focusing on what’s real around you, like naming things you can see, and consider talking to a professional if this persists. Would you like to share more about what’s going on?"  
    elif emotion == 'claustrophobic':
        fallback_response = "I’m sorry you’re feeling claustrophobic—that can be really uncomfortable, and I’m here for you. Try finding an open space to breathe deeply, or focus on a calming image to help ease the feeling. Would you like to talk more about what’s triggering this?"  
    elif emotion == 'phobia':
        fallback_response = "I can see you’re dealing with a phobia, and I’m here to support you—it can be really challenging. Try gradual exposure to what scares you in a safe way, or consider speaking with a therapist for specialized help. Would you like to share more about your phobia?"  
    elif emotion == 'agitated':
        fallback_response = "I’m sorry you’re feeling agitated—that can be really uncomfortable, and I’m here for you. Take a moment to step away and try a calming activity, like sipping herbal tea or doing some light stretching. Would you like to talk more about what’s agitating you?"  
    elif emotion == 'tense':
        fallback_response = "I can tell you’re feeling tense, and I’m here to support you—it’s tough to carry that tension. Try a quick relaxation technique, like rolling your shoulders or focusing on slow breaths, to help your body unwind. Would you like to share what’s making you feel this way?"  
    elif emotion == 'distracted':
        fallback_response = "I’m sorry you’re feeling distracted—it can be frustrating when you can’t focus, and I’m here for you. Try creating a quiet space and setting a timer for a short, focused task to help you get back on track. Would you like to talk more about what’s distracting you?"  
    elif emotion == 'disoriented':
        fallback_response = "I can see you’re feeling disoriented, and I’m here to help—it can be unsettling to feel this way. Take a moment to sit down, sip some water, and focus on something familiar to ground yourself. Would you like to share more about what’s going on?"  
    elif emotion == 'alienated':
        fallback_response = "I’m sorry you’re feeling alienated—that can be really isolating, and I’m here for you. Try reaching out to a community or group that shares your interests to help you feel more connected. Would you like to talk more about what’s making you feel alienated?"  
    elif emotion == 'misunderstood':
        fallback_response = "I can tell you’re feeling misunderstood, and I’m here to support you—it’s hard when you feel unheard. Try expressing your thoughts in a different way, like writing them down, or talking to someone who listens well. Would you like to share more about your experience?"  
    elif emotion == 'judged':
        fallback_response = "I’m sorry you’re feeling judged—that can be really hurtful, and I’m here for you. Remind yourself that your worth isn’t defined by others’ opinions, and focus on spending time with people who accept you as you are. Would you like to talk more about what happened?"  
    elif emotion == 'excluded':
        fallback_response = "I can see you’re feeling excluded, and I’m here to support you—it’s tough to feel left out. Try initiating a small gathering with people you trust, or join a group where you feel welcomed. Would you like to share more about what’s making you feel this way?"  
    elif emotion == 'rejected':
        fallback_response = "I’m sorry you’re feeling rejected—that can be really painful, and I’m here for you. Take some time to nurture yourself with an activity you love, and remember that rejection doesn’t define your value. Would you like to talk more about what happened?"  
    elif emotion == 'betrayed':
        fallback_response = "I can tell you’re feeling betrayed, and I’m here to support you—it’s so hard to experience that. Give yourself permission to feel your emotions, and consider setting boundaries with the person who hurt you. Would you like to share more about what happened?"  
    elif emotion == 'abused':
        fallback_response = "I’m so sorry you’re feeling abused—it’s not okay, and I’m here for you. Please prioritize your safety by reaching out to a trusted person or a helpline, like the National Domestic Violence Hotline at 1-800-799-7233. Can we talk more about how to get you support?"  
    elif emotion == 'neglected':
        fallback_response = "I’m sorry you’re feeling neglected—that can be really hurtful, and I’m here for you. Try communicating your needs to someone you trust, or focus on self-care to remind yourself of your worth. Would you like to talk more about what’s going on?"  
    elif emotion == 'unsafe':
        fallback_response = "I’m really concerned that you’re feeling unsafe, and I’m here for you—your safety is so important. If you’re in immediate danger, please call emergency services like 911, or find a safe place to go. Can we talk more about how to keep you safe?"  
    elif emotion == 'vulnerable':
        fallback_response = "I can tell you’re feeling vulnerable, and I’m here to support you—it’s okay to feel this way sometimes. Try surrounding yourself with people or things that make you feel secure, like a close friend or a cozy blanket. Would you like to share more about how you’re feeling?"  
    elif emotion == 'exposed':
        fallback_response = "I’m sorry you’re feeling exposed—that can be really uncomfortable, and I’m here for you. Take a moment to do something that helps you feel safe, like stepping away from the situation or talking to someone you trust. Would you like to talk more about what’s going on?"  
    elif emotion == 'overstimulated':
        fallback_response = "I can see you’re feeling overstimulated, and I’m here to help—it can be a lot to handle. Try finding a quiet space and focusing on one calming sensation, like listening to soft music or closing your eyes for a moment. Would you like to share more about what’s overwhelming you?"  
    elif emotion == 'sensory overload':
        fallback_response = "I’m sorry you’re experiencing sensory overload—that can be really intense, and I’m here for you. Step into a calm environment if you can, and reduce stimulation by dimming lights or using noise-canceling headphones. Would you like to talk more about what’s triggering this?"  
    elif emotion == 'addiction':
        fallback_response = "I can tell you’re struggling with addiction, and I’m here for you—it’s a brave step to acknowledge this. Consider reaching out to a support group like AA or NA, or talking to a professional for guidance on recovery. Would you like to share more about what you’re going through?"  
    elif emotion == 'withdrawal':
        fallback_response = "I’m sorry you’re going through withdrawal—it can be really challenging, and I’m here to support you. Make sure to stay hydrated and consider seeking medical advice to manage your symptoms safely. Would you like to talk more about what you’re experiencing?"  
    elif emotion == 'eating disorder':
        fallback_response = "I’m here for you—I’m so sorry you’re struggling with an eating disorder, and it’s brave to share this. Please consider talking to a professional, like a therapist or dietitian, who can help you with a recovery plan. Would you like to talk more about how you’re feeling?"  
    elif emotion == 'anorexia':
        fallback_response = "I’m really sorry you’re dealing with anorexia—it’s a tough struggle, and I’m here for you. Reaching out to a healthcare provider or a support group can be a helpful step toward recovery, and I’d encourage you to do so. Would you like to share more about your experience?"  
    elif emotion == 'bulimia':
        fallback_response = "I can tell you’re struggling with bulimia, and I’m here to support you—it’s a lot to handle. A therapist or support group can offer strategies to help you heal, so please consider reaching out for professional help. Would you like to talk more about what you’re going through?"  
    elif emotion == 'binge eating':
        fallback_response = "I’m sorry you’re dealing with binge eating—it can be really overwhelming, and I’m here for you. Try keeping a food journal to understand your triggers, and consider speaking with a therapist for support. Would you like to share more about your experience?"  
    elif emotion == 'body dysmorphia':
        fallback_response = "I can see you’re struggling with body dysmorphia, and I’m here for you—it’s tough to feel this way about yourself. A therapist can help you work through these feelings with techniques like CBT, so I’d encourage you to reach out. Would you like to talk more about how you’re feeling?"  
    elif emotion == 'gender dysphoria':
        fallback_response = "I’m sorry you’re experiencing gender dysphoria—it can be really challenging, and I’m here for you. Connecting with a supportive community or a therapist who specializes in gender identity can make a big difference. Would you like to share more about what you’re going through?"  
    elif emotion == 'sexual orientation struggles':
        fallback_response = "I can tell you’re struggling with your sexual orientation, and I’m here to support you—it’s okay to take your time with this. Talking to a trusted friend or joining an LGBTQ+ support group might help you feel more understood. Would you like to talk more about your feelings?"  
    elif emotion == 'cultural pressure':
        fallback_response = "I’m sorry you’re feeling cultural pressure—that can be really heavy, and I’m here for you. Try setting boundaries that honor your own values while finding small ways to connect with your culture on your terms. Would you like to share more about what’s been challenging?"  
    elif emotion == 'racial discrimination':
        fallback_response = "I’m so sorry you’re experiencing racial discrimination—it’s not okay, and I’m here for you. Consider connecting with a community that supports you, or speaking to someone who can help address the situation. Would you like to talk more about what happened?"  
    elif emotion == 'microaggressions':
        fallback_response = "I can tell you’re dealing with microaggressions, and I’m here to support you—they can be really hurtful over time. Try educating those around you if it feels safe, or find a supportive space where you feel valued. Would you like to share more about your experience?"  
    elif emotion == 'stereotyped':
            fallback_response = "I’m sorry you’re feeling stereotyped—that can be really frustrating, and I’m here for you. Focus on connecting with people who see you for who you truly are, and consider sharing your perspective to challenge those stereotypes. Would you like to talk more about what happened?"  
    elif emotion == 'language barrier':
        fallback_response = "I can see you’re struggling with a language barrier, and I’m here to help—it can be isolating. Try using simple phrases or a translation app to communicate, and be patient with yourself as you learn. Would you like to talk more about what’s been challenging?"  
    elif emotion == 'homesick':
        fallback_response = "I’m sorry you’re feeling homesick—it can be really tough to be far from home, and I’m here for you. Try creating a small reminder of home, like cooking a favorite meal, or calling a loved one to feel more connected. Would you like to share more about what you miss?"  
    elif emotion == 'culture shock':
        fallback_response = "I can tell you’re experiencing culture shock, and I’m here to support you—it can be disorienting. Take it slow by learning one new thing about the culture each day, and find someone local to guide you. Would you like to talk more about what’s been surprising for you?"  
    elif emotion == 'immigration stress':
        fallback_response = "I’m sorry you’re dealing with immigration stress—it can be so overwhelming, and I’m here for you. Try connecting with a support organization for immigrants, and take it one step at a time with paperwork or adjustments. Would you like to talk more about what’s been stressful?"  
    elif emotion == 'legal issues':
        fallback_response = "I can see you’re dealing with legal issues, and I’m here to support you—that can be really stressful. Consider reaching out to a trusted lawyer or legal aid service for guidance, and take care of yourself during this process. Would you like to share more about what’s going on?"  
    elif emotion == 'court case stress':
        fallback_response = "I’m sorry you’re feeling stressed about a court case—it’s a lot to handle, and I’m here for you. Try preparing with your lawyer and focusing on self-care, like taking short walks, to manage the stress. Would you like to talk more about what’s happening?"  
    elif emotion == 'debt stress':
        fallback_response = "I can tell you’re stressed about debt, and I’m here to support you—it can feel so heavy. Try creating a small repayment plan or speaking with a financial counselor to ease the burden. Would you like to talk more about your situation?"  
    elif emotion == 'bankruptcy':
        fallback_response = "I’m sorry you’re dealing with bankruptcy—it’s a tough situation, and I’m here for you. Reach out to a financial advisor to understand your options, and focus on small steps to rebuild, like setting a budget. Would you like to share more about what’s going on?"  
    elif emotion == 'housing issues':
        fallback_response = "I can see you’re struggling with housing issues, and I’m here to support you—that can be so stressful. Look into local resources like housing assistance programs, and try to create a temporary plan to feel secure. Would you like to talk more about your situation?"  
    elif emotion == 'homeless':
        fallback_response = "I’m so sorry you’re experiencing homelessness—I’m here for you, and you’re not alone. Please reach out to a local shelter or organization for immediate support, and let’s talk about how to get you to a safe place. Would you like help finding resources?"  
    elif emotion == 'relationship abuse':
        fallback_response = "I’m really sorry you’re experiencing relationship abuse—it’s not okay, and I’m here for you. Your safety is the priority, so please consider contacting a helpline like the National Domestic Violence Hotline at 1-800-799-7233. Can we talk more about how to get you support?"  
    elif emotion == 'gaslighting':
        fallback_response = "I can tell you’re dealing with gaslighting, and I’m here to support you—it can be so confusing and hurtful. Trust your own feelings and consider talking to a therapist or a trusted friend to help you process what’s happening. Would you like to share more about your experience?"  
    elif emotion == 'manipulation':
        fallback_response = "I’m sorry you’re experiencing manipulation—it’s not okay, and I’m here for you. Try setting clear boundaries with the person, and seek support from someone who can help you navigate this situation. Would you like to talk more about what’s going on?"  
    elif emotion == 'toxic relationship':
        fallback_response = "I can see you’re in a toxic relationship, and I’m here to support you—it’s hard to be in that space. Consider taking a step back to prioritize your well-being, and talk to a trusted person or therapist about your next steps. Would you like to share more about your situation?"  
    elif emotion == 'friendship betrayal':
        fallback_response = "I’m sorry you’re dealing with a friendship betrayal—that can be so painful, and I’m here for you. Give yourself time to process your feelings, and consider talking to your friend about how their actions affected you. Would you like to share more about what happened?"  
    elif emotion == 'sibling rivalry':
        fallback_response = "I can tell you’re experiencing sibling rivalry, and I’m here to support you—it can be really challenging. Try having an open conversation with your sibling to understand each other’s perspectives, or focus on your own space for a while. Would you like to talk more about what’s going on?"  
    elif emotion == 'parental expectations':
        fallback_response = "I’m sorry you’re feeling pressure from parental expectations—it can be so heavy, and I’m here for you. Try communicating your own goals and boundaries with them, and focus on what makes you feel fulfilled. Would you like to share more about what they expect?"  
    elif emotion == 'caregiving stress':
        fallback_response = "I can see you’re dealing with caregiving stress, and I’m here to support you—it’s a lot to take on. Make sure to carve out time for yourself, even if it’s just a few minutes to rest, and consider joining a caregiver support group. Would you like to talk more about your responsibilities?"  
    elif emotion == 'elderly care stress':
        fallback_response = "I’m sorry you’re feeling stressed about elderly care—it’s a big responsibility, and I’m here for you. Try reaching out to community resources for help, and make time for self-care to recharge. Would you like to share more about what’s been challenging?"  
    elif emotion == 'childcare stress':
        fallback_response = "I can tell you’re stressed about childcare, and I’m here to support you—it can be overwhelming. Try creating a simple routine for both you and the kids, and ask for help from family or friends when you need a break. Would you like to talk more about what’s going on?"  
    elif emotion == 'academic pressure':
        fallback_response = "I’m sorry you’re feeling academic pressure—that can be really intense, and I’m here for you. Break your study tasks into smaller chunks, and make sure to take breaks to avoid burnout. Would you like to talk more about what’s been overwhelming at school?"  
    elif emotion == 'exam stress':
        fallback_response = "I can see you’re stressed about exams, and I’m here to support you—it’s a lot to handle. Try creating a study schedule with short, focused sessions, and practice deep breathing to stay calm during the exam. Would you like to share more about your exams?"  
    elif emotion == 'failing grades':
        fallback_response = "I’m sorry you’re dealing with failing grades—that can be really discouraging, and I’m here for you. Reach out to a teacher or tutor for support, and focus on small improvements rather than perfection. Would you like to talk more about what’s been challenging?"  
    elif emotion == 'school bullying':
        fallback_response = "I’m so sorry you’re experiencing school bullying—it’s not okay, and I’m here for you. Tell a trusted adult, like a teacher or parent, about what’s happening, and focus on activities that make you feel safe and happy. Would you like to share more about what’s going on?"  
    elif emotion == 'peer pressure':
        fallback_response = "I can tell you’re dealing with peer pressure, and I’m here to support you—it can be really tough. Practice saying no in a way that feels comfortable to you, and spend time with friends who respect your choices. Would you like to talk more about the situation?"  
    elif emotion == 'social media pressure':
        fallback_response = "I’m sorry you’re feeling social media pressure—that can be so overwhelming, and I’m here for you. Try taking a break from social media and focusing on real-life connections that make you feel good about yourself. Would you like to talk more about what’s been affecting you?"  
    elif emotion == 'online harassment':
        fallback_response = "I’m so sorry you’re experiencing online harassment—it’s not okay, and I’m here for you. Report the behavior to the platform, and consider taking a break from online spaces while surrounding yourself with supportive people. Would you like to talk more about what happened?"  
    elif emotion == 'cyberbullying':
        fallback_response = "I can tell you’re dealing with cyberbullying, and I’m here to support you—it’s really tough to face this. Save evidence of the bullying, report it to the platform or a trusted adult, and focus on offline activities that bring you joy. Would you like to share more about what’s going on?"  
    elif emotion == 'technology overload':
        fallback_response = "I’m sorry you’re feeling overwhelmed by technology—it can be a lot, and I’m here for you. Set specific times to unplug, like during meals or before bed, and engage in a screen-free activity like reading or walking. Would you like to talk more about your tech use?"  
    elif emotion == 'screen time stress':
        fallback_response = "I can see you’re stressed about screen time, and I’m here to support you—it’s easy to feel overwhelmed by screens. Try scheduling screen-free hours each day, and replace that time with something relaxing, like a hobby or spending time outdoors. Would you like to share more about your routine?"  
    elif emotion == 'pandemic stress':
        fallback_response = "I’m sorry you’re feeling stressed about the pandemic—it’s been a challenging time, and I’m here for you. Focus on what you can control, like maintaining a routine, and stay connected with loved ones for support. Would you like to talk more about how it’s affecting you?"  
    elif emotion == 'quarantine stress':
        fallback_response = "I can tell you’re feeling stressed about quarantine, and I’m here to support you—it can be really isolating. Create a small daily structure, like setting aside time for a hobby, and reach out to friends virtually to stay connected. Would you like to share more about what’s been hard?"  
    elif emotion == 'health anxiety':
        fallback_response = "I’m sorry you’re dealing with health anxiety—it can be so overwhelming, and I’m here for you. Try limiting how often you research symptoms online, and focus on a grounding activity like deep breathing when worries arise. Would you like to talk more about your concerns?"  
    elif emotion == 'chronic illness':
        fallback_response = "I can see you’re dealing with a chronic illness, and I’m here to support you—it’s a lot to manage. Make sure to communicate with your healthcare team, and carve out time for rest and activities that bring you comfort. Would you like to share more about your experience?"  
    elif emotion == 'disability challenges':
        fallback_response = "I’m sorry you’re facing challenges with a disability—I’m here for you, and you’re not alone. Connect with a support group for people with similar experiences, and advocate for accommodations that can help you. Would you like to talk more about what’s been difficult?"  
    elif emotion == 'medical procedure stress':
        fallback_response = "I can tell you’re stressed about a medical procedure, and I’m here to support you—it’s normal to feel this way. Ask your doctor questions to feel more prepared, and bring a comforting item, like a favorite playlist, to the procedure. Would you like to share more about what’s coming up?"  
    elif emotion == 'surgery recovery':
        fallback_response = "I’m sorry you’re going through surgery recovery—it can be tough, and I’m here for you. Follow your doctor’s advice, rest as much as you can, and ask for help from loved ones when you need it. Would you like to talk more about how you’re feeling?"  
    elif emotion == 'grieving pet':
        fallback_response = "I’m so sorry you’re grieving your pet—it’s a deep loss, and I’m here for you. Create a small memorial, like planting a flower in their memory, to honor them while you process your grief. Would you like to share more about your pet?"  
    elif emotion == 'pet loss':
        fallback_response = "I can tell you’re grieving the loss of your pet, and I’m here to support you—it’s so hard to lose a companion. Spend time looking at photos or writing about your favorite memories with them to help you heal. Would you like to talk more about what they meant to you?"  
    elif emotion == 'aging concerns':
        fallback_response = "I’m sorry you’re feeling concerned about aging—it can bring up a lot of emotions, and I’m here for you. Focus on what you enjoy about this stage of life, and try staying active with gentle exercises to feel your best. Would you like to share more about your worries?"  
    elif emotion == 'retirement stress':
        fallback_response = "I can see you’re stressed about retirement, and I’m here to support you—it’s a big transition. Create a new routine that includes hobbies or volunteering to give your days purpose, and connect with others who are also retired. Would you like to talk more about what’s on your mind?"  
    elif emotion == 'empty nest syndrome':
        fallback_response = "I’m sorry you’re experiencing empty nest syndrome—it can feel so quiet, and I’m here for you. Try rediscovering hobbies or interests you’ve always wanted to explore, and stay connected with your kids through calls or visits. Would you like to share more about how you’re feeling?"  
    elif emotion == 'midlife crisis':
        fallback_response = "I can tell you’re going through a midlife crisis, and I’m here to support you—it’s a challenging time. Reflect on what truly matters to you now, and set a small, meaningful goal to bring a sense of purpose. Would you like to talk more about what’s been on your mind?"  
    elif emotion == 'spirituality crisis':
        fallback_response = "I’m sorry you’re experiencing a spirituality crisis—it can feel unsettling, and I’m here for you. Explore your beliefs by journaling or talking to a spiritual advisor, and give yourself permission to question without pressure. Would you like to share more about your journey?"  
    elif emotion == 'faith doubts':
        fallback_response = "I can see you’re having doubts about your faith, and I’m here to support you—it’s okay to question. Try talking to someone in your faith community or reading about others’ experiences to help you reflect. Would you like to talk more about what’s been challenging?"  
    elif emotion == 'existential crisis':
        fallback_response = "I’m sorry you’re going through an existential crisis—it can feel overwhelming, and I’m here for you. Try exploring your thoughts through journaling or philosophy, and focus on small acts of meaning, like helping others. Would you like to share more about what’s on your mind?"  
    elif emotion == 'lack of purpose':
        fallback_response = "I can tell you’re feeling a lack of purpose, and I’m here to support you—it’s tough to feel this way. Start by trying a new activity or volunteering to connect with something meaningful, and give yourself time to discover what fulfills you. Would you like to talk more about what you’re searching for?"  
    elif emotion == 'climate anxiety':
        fallback_response = "I’m sorry you’re feeling climate anxiety—it’s a heavy concern, and I’m here for you. Channel your energy into a small, positive action, like reducing waste or joining a local environmental group, to feel more empowered. Would you like to talk more about your worries?"  
    elif emotion == 'environmental stress':
        fallback_response = "I can see you’re stressed about environmental issues, and I’m here to support you—it’s a lot to carry. Focus on sustainable habits you can adopt, like conserving energy, and connect with others who share your concerns. Would you like to share more about what’s been on your mind?"  
    elif emotion == 'natural disaster stress':
        fallback_response = "I’m so sorry you’re dealing with stress from a natural disaster—it’s incredibly tough, and I’m here for you. Make sure you’re safe first, and reach out to local resources for support while taking time to process your emotions. Would you like to talk more about what happened?"  
    elif emotion == 'war trauma':
        fallback_response = "I’m deeply sorry you’re experiencing war trauma—it’s an immense burden, and I’m here for you. Please consider speaking with a trauma therapist, and focus on grounding techniques like breathing exercises to help manage your feelings. Would you like to share more about your experience?"  
    elif emotion == 'political stress':
        fallback_response = "I can tell you’re feeling stressed about political issues, and I’m here to support you—it can be overwhelming. Try limiting your news intake and focusing on actions you can take, like voting or volunteering, to feel more in control. Would you like to talk more about what’s been affecting you?"  
    elif emotion == 'disillusioned':
        fallback_response = "I’m sorry you’re feeling disillusioned—it can be tough when things don’t align with your expectations, and I’m here for you. Take some time to reflect on what still inspires you, perhaps by revisiting a passion or value that feels true to you. Would you like to share more about what’s left you feeling this way?"  
    elif emotion == 'hopeless':
        fallback_response = "I can sense you’re feeling hopeless, and I’m here to support you—it’s really hard to feel this way. Try focusing on one small, positive step you can take today, like reaching out to a friend, to remind yourself that you’re not alone. Would you like to talk more about what’s making you feel hopeless?"  
    elif emotion == 'despair':
        fallback_response = "I’m so sorry you’re feeling despair—it’s a heavy burden, and I’m here for you. Sometimes, connecting with nature, like sitting under a tree or listening to the wind, can offer a moment of peace—would you like to try that? Can you tell me more about what’s going on?"  
    elif emotion == 'numb':
        fallback_response = "I can tell you’re feeling numb, and I’m here to support you—it’s tough when emotions feel so distant. Try engaging your senses, like holding a warm cup of tea or listening to a favorite song, to gently reconnect with yourself. Would you like to share more about how you’re feeling?"  
    elif emotion == 'detached':
        fallback_response = "I’m sorry you’re feeling detached—it can be unsettling to feel so disconnected, and I’m here for you. Maybe try a grounding exercise, like feeling the texture of an object in your hand, to bring yourself back to the present moment. Would you like to talk more about what’s making you feel this way?"  
    elif emotion == 'restless':
        fallback_response = "I can see you’re feeling restless, and I’m here to help—it’s hard when you can’t settle. Try channeling that energy into a short, creative task, like doodling or writing a quick poem, to help you feel more centered. Would you like to share more about what’s got you feeling restless?"  
    elif emotion == 'impatient':
        fallback_response = "I’m sorry you’re feeling impatient—it can be frustrating to wait, and I’m here for you. Try focusing on the present by naming three things you can see around you, which can help ease the tension of waiting. Would you like to talk more about what’s making you feel this way?"  
    elif emotion == 'frantic':
        fallback_response = "I can tell you’re feeling frantic, and I’m here to support you—it’s overwhelming to feel so rushed. Pause for a moment and try sipping a glass of cold water slowly to help slow your pace and calm your mind. Would you like to share more about what’s got you so frantic?"  
    elif emotion == 'exhausted':
        fallback_response = "I’m sorry you’re feeling exhausted—it’s tough when your energy is so low, and I’m here for you. Give yourself permission to rest, even if it’s just closing your eyes for five minutes while listening to soft sounds, like rain. Would you like to talk more about what’s been draining you?"  
    elif emotion == 'burnout':
        fallback_response = "I can see you’re experiencing burnout, and I’m here to support you—it’s a lot to handle when you’re so overwhelmed. Try setting a small boundary, like taking a 10-minute break with no screens, to give yourself a moment to recharge. Would you like to share more about what’s been exhausting you?"  
    elif emotion == 'sleep issues':
        fallback_response = "I’m sorry you’re dealing with sleep issues—that can be so frustrating, and I’m here for you. Create a calming pre-sleep routine, like dimming the lights and reading a light book, to help signal to your body it’s time to rest. Would you like to talk more about what’s keeping you up?"  
    elif emotion == 'insomnia':
        fallback_response = "I can tell you’re struggling with insomnia, and I’m here to support you—it’s tough when sleep won’t come. Try a wind-down ritual, like sipping chamomile tea and avoiding screens an hour before bed, to help your mind relax. Would you like to share more about what’s been affecting your sleep?"  
    elif emotion == 'nightmares':
        fallback_response = "I’m so sorry you’re having nightmares—they can be really distressing, and I’m here for you. Before bed, try imagining a peaceful scene, like a quiet beach, to help your mind focus on calm images as you drift off. Would you like to talk more about what you’ve been dreaming about?"  
    elif emotion == 'trauma':
        fallback_response = "I’m deeply sorry you’re dealing with trauma—it’s a heavy weight, and I’m here for you. Consider finding a safe space to express your feelings, like through journaling or speaking with a trauma-informed therapist. Would you like to share more about what you’re going through?"  
    elif emotion == 'ptsd':
        fallback_response = "I can see you’re struggling with PTSD, and I’m here to support you—it’s incredibly challenging. Try creating a comforting routine, like having a cozy blanket nearby, to help you feel safe when memories surface. Would you like to talk more about what’s been triggering for you?"  
    elif emotion == 'flashbacks':
        fallback_response = "I’m so sorry you’re experiencing flashbacks—they can be really overwhelming, and I’m here for you. When one happens, try focusing on your surroundings by naming five things you can touch to bring yourself back to the present. Would you like to share more about what’s been coming up for you?"  
    elif emotion == 'panic attack':
        fallback_response = "I can tell you’re dealing with a panic attack, and I’m here to help—it’s so hard to go through that. Try holding an ice cube or splashing your face with cold water to help your body reset and calm down. Would you like to talk more about what triggered this?"  
    elif emotion == 'social anxiety':
        fallback_response = "I’m sorry you’re feeling social anxiety—it can make interactions so tough, and I’m here for you. Before a social event, try practicing a simple affirmation, like ‘I am enough,’ to boost your confidence. Would you like to share more about what’s making you anxious?"  
    elif emotion == 'public speaking fear':
        fallback_response = "I can see you’re nervous about public speaking, and I’m here to support you—it’s a common fear. Picture your audience as friends, and practice your speech in front of a mirror to build familiarity and ease. Would you like to talk more about your upcoming talk?"  
    elif emotion == 'performance anxiety':
        fallback_response = "I’m sorry you’re experiencing performance anxiety—it can be so stressful, and I’m here for you. Try visualizing a successful performance in detail, like hearing applause, to help shift your focus to a positive outcome. Would you like to share more about what you’re preparing for?"  
    elif emotion == 'imposter syndrome':
        fallback_response = "I can tell you’re struggling with imposter syndrome, and I’m here to support you—it’s hard to feel like you don’t belong. Make a list of your achievements, big or small, to remind yourself of your worth and all you’ve accomplished. Would you like to talk more about what’s making you doubt yourself?"  
    elif emotion == 'perfectionism':
        fallback_response = "I’m sorry you’re dealing with perfectionism—it can be exhausting to chase that standard, and I’m here for you. Try setting a ‘good enough’ goal for a task, and celebrate completing it without overthinking the details. Would you like to share more about what you’re working on?"  
    elif emotion == 'procrastination':
        fallback_response = "I can see you’re struggling with procrastination, and I’m here to help—it’s easy to put things off. Break your task into tiny steps and start with just the first one, like writing a single sentence, to build momentum. Would you like to talk more about what you’re avoiding?"  
    elif emotion == 'time management stress':
        fallback_response = "I’m sorry you’re feeling stressed about time management—it can feel chaotic, and I’m here for you. Try using a simple timer method, like working for 25 minutes and taking a 5-minute break, to help you stay focused and organized. Would you like to share more about your schedule?" 
    elif emotion == 'decision fatigue':
        fallback_response = "I can tell you’re dealing with decision fatigue, and I’m here to support you—it’s tough when choices feel overwhelming. For now, simplify by picking just one small decision, like what to eat for lunch, and let the rest wait until you feel clearer. Would you like to talk more about what’s been piling up?"  
    elif emotion == 'trust issues':
        fallback_response = "I’m sorry you’re struggling with trust issues—it can be hard to feel safe with others, and I’m here for you. Start by building trust in small ways, like sharing something minor with someone reliable, to slowly rebuild that sense of safety. Would you like to share more about what’s been challenging?"
    elif emotion == 'attachment issues':
        fallback_response = "I can see you’re dealing with attachment issues, and I’m here to support you—it’s tough to navigate those feelings. Reflect on your relationships by noting what feels safe and what doesn’t, which can help you understand your needs better. Would you like to talk more about your experiences?"  
    elif emotion == 'abandonment fear':
        fallback_response = "I’m sorry you’re feeling a fear of abandonment—it can be so unsettling, and I’m here for you. Try building a support network by nurturing a few close relationships where you feel valued, which can help ease that fear. Would you like to share more about what’s been triggering this?"  
    elif emotion == 'jealousy':
        fallback_response = "I can tell you’re feeling jealous, and I’m here to support you—it’s a tough emotion to handle. Focus on your own strengths by writing down three things you’re proud of about yourself, which can help shift your perspective. Would you like to talk more about what’s making you feel jealous?"  
    elif emotion == 'insecurity':
        fallback_response = "I’m sorry you’re feeling insecure—it’s hard to doubt yourself, and I’m here for you. Try wearing or holding something that makes you feel confident, like a favorite accessory, to give yourself a small boost. Would you like to share more about what’s been making you feel this way?"  
    elif emotion == 'self-doubt':
        fallback_response = "I can see you’re struggling with self-doubt, and I’m here to support you—it’s tough to feel uncertain about yourself. Ask a trusted friend to share a quality they admire in you, which can help you see yourself through their eyes. Would you like to talk more about what’s causing your doubt?"  
    elif emotion == 'identity crisis':
        fallback_response = "I’m sorry you’re going through an identity crisis—it can feel so confusing, and I’m here for you. Try exploring different aspects of yourself through creative outlets, like painting or music, to help you reconnect with who you are. Would you like to share more about what’s been on your mind?"  
    elif emotion == 'generational trauma':
        fallback_response = "I can tell you’re dealing with generational trauma, and I’m here to support you—it’s a heavy legacy to carry. Consider exploring your family history with a therapist to understand its impact, which can help you start healing. Would you like to share more about what you’ve noticed?"  
    elif emotion == 'family conflict':
        fallback_response = "I’m sorry you’re experiencing family conflict—it can be so stressful, and I’m here for you. Try finding a neutral space to have a calm conversation with your family, focusing on listening as much as speaking. Would you like to talk more about what’s been happening?"  
    elif emotion == 'divorce stress':
        fallback_response = "I can see you’re dealing with divorce stress, and I’m here to support you—it’s a lot to go through. Create a self-care ritual, like taking a warm bath each evening, to give yourself some comfort during this transition. Would you like to share more about what’s been challenging?"  
    elif emotion == 'custody battle stress':
        fallback_response = "I’m so sorry you’re going through a custody battle—it’s incredibly tough, and I’m here for you. Focus on creating a stable routine for yourself and your kids, which can provide some comfort amidst the uncertainty. Would you like to talk more about what’s been happening?"  
    elif emotion == 'financial insecurity':
        fallback_response = "I can tell you’re feeling financial insecurity, and I’m here to support you—it’s a heavy worry. Start by tracking your expenses for a week to see where small changes can be made, which might help you feel more in control. Would you like to share more about your situation?"  
    elif emotion == 'job insecurity':
        fallback_response = "I’m sorry you’re feeling job insecurity—it can be so unsettling, and I’m here for you. Update your resume or learn a small new skill online to boost your confidence and feel more prepared for any changes. Would you like to talk more about what’s been worrying you?"  
    elif emotion == 'unemployment stress':
        fallback_response = "I can see you’re stressed about unemployment, and I’m here to support you—it’s a tough spot to be in. Set a small daily goal, like applying to one job or networking with one person, to keep moving forward without feeling overwhelmed. Would you like to share more about your job search?" 
    elif emotion == 'career change stress':
        fallback_response = "I’m sorry you’re feeling stressed about a career change—it’s a big shift that can bring up a lot of uncertainty, and I’m here for you to help you navigate this transition. Research one aspect of your new field, like joining a related online community or reading about someone’s success story, to help you feel more connected and confident in your decision. Would you like to talk more about your career goals and what’s been the most challenging part of this change for you?"
    elif emotion == 'greeting_hi':
        fallback_response = "Hi there! I’m so glad you’re here—let’s chat about whatever’s on your mind, whether it’s something exciting or something you need support with. How can I support you today?"
    elif emotion == 'greeting_hello':
        fallback_response = "Hello! It’s great to hear from you—I’m here to help with anything you’d like to talk about, from challenges to happy moments. What’s on your mind?"
    elif emotion == 'greeting_how_are_you':
        fallback_response = "I’m doing great, thanks for asking—how about you? I’m here to listen and support you with whatever you’re feeling, no matter how big or small. What’s going on?"
    elif emotion == 'greeting_hey':
        fallback_response = "Hey! I’m happy to chat with you—thanks for reaching out, it means a lot to connect with you. What would you like to talk about today?"
    elif emotion == 'greeting_good_morning':
        fallback_response = "Good morning to you too! I hope your day’s off to a great start—I’m here to help with anything on your mind, whether it’s a challenge or a goal for the day. How can I support you today?"
    elif emotion == 'farewell_bye':
        fallback_response = "Bye for now! I’ve really enjoyed talking with you—feel free to come back anytime you need to chat, as I’ll always be here for you. Take care!"
    elif emotion == 'farewell_thanks':
        fallback_response = "You’re welcome—I’m glad I could help! Thanks for chatting with me, and I’m here if you need me again, anytime. Have a great day!"
    elif emotion == 'farewell_thank_you':
        fallback_response = "My pleasure—I’m so happy I could be there for you! Thank you for sharing, and I’ll be here whenever you’d like to talk again, so don’t hesitate to reach out. Take care!"
    elif emotion == 'farewell_oh':
        fallback_response = "I hear you—it sounds like you’re reflecting on something, and I’m here if you’d like to explore it more another time. Thanks for chatting, and feel free to return anytime—I’ll be waiting! Take care!"
    elif emotion == 'farewell_see_you':
        fallback_response = "See you soon! I’ve really enjoyed our conversation, and I’ll be here whenever you’re ready to chat again, so don’t be a stranger. Take care until then!"
    elif emotion == 'music_for_stress':
        fallback_response = "Try listening to Beethoven’s Moonlight Sonata—it’s a calming classical piece to help ease your stress."
    elif emotion == 'face_care':
        fallback_response = "Try using aloe vera gel as a natural moisturizer to soothe stressed skin."
    elif emotion == 'hair_growth' :
        fallback_response = "Massage your scalp with coconut oil weekly—it can nourish your hair follicles and promote growth."
    elif emotion == 'hair_care_tips' :
        fallback_response = "Massage your scalp with coconut oil weekly and use natural products like aloevera,onion,curry leaves etc. to make dyes,masks,shampoos etc., for hair growth,instead of replying on artificial products."
    elif emotion == 'stress_relief_activity':
        fallback_response  = "Consider going for a short walk—it can help clear your mind and reduce stress."
    elif emotion == 'sleep_improvement':
        fallback_response = "Create a calming bedtime routine—try dimming the lights and avoiding screens an hour before bed."
    elif emotion == 'hydration_reminder':
        fallback_response = "Yes, staying hydrated is key for your well-being! Aim for at least 8 glasses of water a day."
    elif emotion == 'morning_routine':
        fallback_response = "Start your day with a glass of water, some light stretching, and a healthy breakfast."
    elif emotion == 'evening_relaxation':
        fallback_response = "Unwind with a warm bath and some chamomile tea—it can help you de-stress."    
    elif emotion == 'energy_boost':
        fallback_response = "Try a quick 5-minute stretch or a brisk walk to boost your energy levels."    
    elif emotion == 'focus_improvement':
        fallback_response = "Take short breaks every 25 minutes using the Pomodoro technique—it can improve your concentration."
    elif emotion == 'nail_care':
        fallback_response = "Keep your nails clean and moisturized—apply cuticle oil like Sally Hansen Vitamin E Nail & Cuticle Oil."    
    elif emotion == 'dry_hands':
        fallback_response = "Use a rich hand cream like O’Keeffe’s Working Hands to hydrate and repair dry skin."    
    elif emotion == 'foot_care':
        fallback_response = "Soak your feet in warm water with Epsom salt, then moisturize with a cream like CeraVe Renewing SA Foot Cream."    
    elif emotion == 'back_pain_relief':            
        fallback_response = "Try gentle stretching or a warm compress to ease the tension in your back."    
    elif emotion == 'eye_strain':
        fallback_response = "Follow the 20-20-20 rule: every 20 minutes, look at something 20 feet away for 20 seconds."    
    elif emotion == 'healthy_snack':
        fallback_response = "Try a handful of almonds and an apple—they’re nutritious and satisfying."    
    elif emotion == 'mood_booster':
        fallback_response = "Listen to an upbeat song or call a friend—social connection can lift your spirits."    
    elif emotion == 'stress_eating':
        fallback_response = "Swap out snacks for a calming activity like journaling or deep breathing to manage stress."    
    elif emotion == 'sun_protection':
        fallback_response = "Yes, apply a broad-spectrum sunscreen like Neutrogena Ultra Sheer SPF 50 daily to protect your skin."
    elif emotion == 'posture_improvement':
        fallback_response = "Sit up straight and keep your shoulders relaxed—try setting a reminder to check your posture hourly."    
    elif emotion == 'lip_care':
        fallback_response = "Use a hydrating lip balm like Burt’s Bees Beeswax Lip Balm to soothe and moisturize."    
    elif emotion == 'scalp_health':
        fallback_response = "Use a gentle shampoo and massage your scalp with rosemary oil to promote a healthy scalp."
    elif emotion == 'energy_drink_alternative':
        fallback_response = "Instead, try a natural boost with green tea—it has caffeine and antioxidants without the crash."    
    elif emotion == 'mindfulness_practice':
        fallback_response = "Try a 5-minute mindfulness exercise: focus on your breath and notice your surroundings."    
    elif emotion == 'teeth_care':
        fallback_response = "Brush twice daily and floss—consider a whitening toothpaste like Colgate Optic White."    
    elif emotion == 'cold_hands':
        fallback_response = "Warm them up by running them under lukewarm water or wearing cozy gloves."   
    elif emotion == 'stomach_upset':
        fallback_response = "Sip on ginger tea—it can help settle your stomach naturally."    
    elif emotion == 'neck_pain':
        fallback_response = "Gently stretch your neck side to side and apply a warm compress to relax the muscles."    
    elif emotion == 'dry_eyes':
        fallback_response = "Use artificial tears like Systane Ultra Lubricant Eye Drops to relieve dryness."   
    elif emotion == 'healthy_breakfast':
        fallback_response = "Try oatmeal with berries and a drizzle of honey—it’s nutritious and filling."  
    elif emotion == 'stress_journaling':
        fallback_response = "Yes, writing down your thoughts can help you process stress—try journaling for 10 minutes." 
    elif emotion == 'screen_time_break':
        fallback_response = "Take a break—step away for 10 minutes and focus on something offline, like a quick walk."  
    elif emotion == 'hand_relaxation':
        fallback_response = "Try a hand massage with a bit of lavender oil to relax your muscles."  
    elif emotion == 'leg_cramps':
        fallback_response = "Stretch your calves gently and stay hydrated—electrolytes like potassium can help."  
    elif emotion == 'morning_energy':
        fallback_response = "Drink a glass of lemon water and do a quick 5-minute stretch to wake up your body." 
    elif emotion == 'evening_wind_down':
        fallback_response = "Read a book or listen to a calming podcast—it can help you relax before bed."   
    elif emotion == 'dry_skin':
        fallback_response =  "Use a hydrating moisturizer like CeraVe Moisturizing Cream to lock in moisture."    
    elif emotion == 'headache_relief':
        fallback_response = "Drink water and rest in a dark, quiet room—it can help ease your headache."    
    elif emotion == 'foot_odor':
        fallback_response = "Wash your feet daily and use an odor-control powder like Gold Bond Medicated Foot Powder."   
    elif emotion == 'mood_tracking':
        fallback_response = "Yes, tracking your mood can help you understand patterns—try using a journal or app like Daylio."
    else:
        fallback_response = "I’m not sure I understood that. Could you share more about what’s going on? How can I support you today? You can ask about stress relief, self-care tips, or share how you’re feeling."
    return fallback_response

def generate_response_ml(model, tokenizer, user_input, max_length=128):
    is_offensive, message = filter_offensive_input(user_input)
    if is_offensive:
        return message
    prompt = (
        f"You are a compassionate therapist. Provide an empathetic, concise response (2-3 sentences) "
        f"with practical advice. Avoid repetition and focus on emotional support. User: {user_input}"
    )
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    try:
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=60,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=60,
            top_p=0.9,
            temperature=0.6,
            repetition_penalty=1.2
        )
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt):].strip()
        cleaned_response = clean_response(response, user_input)
        if cleaned_response:
            return cleaned_response
        else:
            return generate_response_rule_based(user_input)
    except Exception as e:
        print(f"Warning: Generation failed - {e}")
        return generate_response_rule_based(user_input)
def test_model():
    model_dir = r"C:\Users\hp\Downloads\AI_Chatbot_Mental_Health_Support\output"
    if not os.path.exists(model_dir):
        print(f"Model directory {model_dir} not found. Ensure the model is placed in the correct path.")
        return
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception as e:
        print(f"Failed to load model or tokenizer from {model_dir}. Error: {e}")
        return
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("I am your Emotional Therapist to solve your issues! Type your message below (or type 'exit' or 'quit' to quit):")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye! Take care.")
            break
        if not user_input.strip():
            print("Bot: Please enter a message.")
            continue
        response = generate_response_ml(model, tokenizer, user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    test_model()