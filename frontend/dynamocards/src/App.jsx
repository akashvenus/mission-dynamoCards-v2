import React, {useState} from "react";
import axios from "axios";
import Flashcard from "./Flashcard";

function App(){
  const[youtubeLink,setYoutubeLink] = useState("");
  const[keyConcepts,setKeyConcepts] = useState([]);

  const handleLinkChange = (event) => {
    setYoutubeLink(event.target.value)
  }

  const sendLink = async() => {
    try{
      const response = await axios.post("http://localhost:8000/analyze_video",{
        youtube_link : youtubeLink
      });
      const data = response.data
      if(data.key_concepts && Array.isArray(data.key_concepts)){
        setKeyConcepts(data.key_concepts)
      }
      else{
        console.error("Data does not contain key concepts : ",data)
        setKeyConcepts([])
      }
    }
    catch(err){
      console.log(err)
    }
  }

  const discardFlashCard = (index) => {
    setKeyConcepts(currentConcepts => currentConcepts.filter((_,i) => i !== index))
  }

  return (
    <div className="App">
      <h1>Link to Flash card generator</h1>
      <input className="inputField" type="text" placeholder="Paster youtube link here" value={youtubeLink} onChange={handleLinkChange}/>
      <button onClick={sendLink}>Generate Flashcards</button>

      <div className="flashcardsContainer">
        {keyConcepts.map((concepts,index) => (
          <Flashcard key={index} term={concepts.concept} definition={concepts.definition} onDiscard={() => discardFlashCard(index)} />
        ))}
      </div>
    </div>

  )
}

export default App