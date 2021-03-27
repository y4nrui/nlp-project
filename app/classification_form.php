<?php



?>

<!DOCTYPE html>
<html lang="en">
<head>
  <title>Bootstrap Example</title>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
  <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>

  <style>
      #detect{
        /* background-color:lightblue; */
        background-image: url(assets/img/fakenews.jpg);
        background-size: auto 100%;
        height: 100%;
        position: absolute;
        top: 10;
        left: 0;
        width: 100%;
      }
      
      .detect-form{
          height: 80%;
          width: 50%;
          background-color: white;
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          -ms-transform: translate(-50%, -50%);
          border-radius: 20px;

      }

      .form-group{
          width: 100%;
      }
      textarea {
        resize: none;
        height: auto;
        width: auto;
        
    }
      .testing{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        -ms-transform: translate(-50%, -50%);
        border-radius: 20px;

      }

      h1,p{
          text-align: center;
          font-family: Arial, Helvetica, sans-serif;
      }

      p{
       font-size: medium; 
      }

      .topnav {
            background-color: #333;
            overflow: hidden;
        }

        /* Style the links inside the navigation bar */
        .topnav a {
            float: left;
            color: #f2f2f2;
            text-align: center;
            padding: 14px 16px;
            text-decoration: none;
            font-size: 17px;
        }

        /* Change the color of links on hover */
        .topnav a:hover {
            background-color: #ddd;
            color: black;
        }

        /* Add a color to the active/current link */
        .topnav a.active {
            background-color: blue;
            color: white;
        }

        .times {
            background-color: red;
        }

  </style>


</head>
<body>

    <div class="topnav">
        <a href="index.html">Home</a>
        <a class="active" href="#detect">Detect Fake News</a>
        <a href="summarization_form.php">Summarize News</a>

    </div>

    <section id="detect">

        <div class='detect-form'>
            <h1><b>Detect Fake News</b></h1>
            <p>Input a news article and click on the <i>Detect</i> button to determine if the news is True or Fake. </p>
            <form method="POST" class='testing' action="classification.php">
                <div class="form-group">
                <label for="news">News Text</label>
                <textarea rows="20" cols = "100" class="form-control" name="article"></textarea>
                </div>

                <input type="submit" class="btn btn-default" name="submit" value="Detect">
            </form>
        </div>
        

    </section>






</body>
</html>