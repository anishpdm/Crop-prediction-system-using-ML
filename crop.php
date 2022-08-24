<!DOCTYPE html>
<html lang="en">
<head>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-wEmeIV1mKuiNpC+IOBjI7aAzPcEZeedi5yW5f2yOq55WWLwNGmvvx4Um1vskeMj0" crossorigin="anonymous">

    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>crop prediction</title>
</head>
<body>
<div class="container">

<div class="row">

<div class="col">

<img src="banner.jpg" height="300" width="100%"  alt="Responsive image"></div>
</div>
<div class="row">
<div class="col col-12 col-sm-2 col-md-2 col-lg-2 col-xl-2"></div>

<div class="col col-12 col-sm-8 col-md-8 col-lg-8 col-xl-8">


<form action="" method="post">

<center><h3> Crop Prediction System</h3>
</center><table class="table table-borderless">

<tr>
    <td>Temperature</td>
    <td><input name="temp" type="text" required class="form-control"></td>
</tr>
<tr>
    <td>Humidity</td>
    <td><input name="hum" type="text"required class="form-control"></td>
</tr>
<tr>
    <td>PH</td>
    <td><input name="ph" type="number" min="5" max="7" required class="form-control"></td>
</tr>
<tr>
    <td>Rainfall</td>
    <td><input name="rain" type="text" required class="form-control"></td>
</tr>
<tr>
    <td></td>
    <td><button name="button" class="btn btn-success">SUBMIT</button></td>
</tr>
</table>



</form>





</div>

<div class="col col-12 col-sm-2 col-md-2 col-lg-2 col-xl-2"></div>

</div>



</div>

    
</body>
</html>

<?php
if(isset($_POST['button']))
{
 $getTemperature=$_POST['temp'];
 $getHumidity=$_POST['hum'];
 $getPH=$_POST['ph'];
 $getRainfall=$_POST['rain'];

//  echo $getTemperature;
//  echo $getHumidity;
//  echo $getPH;
//  echo $getRainfall;
 $result= file_get_contents("http://127.0.0.1:5000/api?temperature=$getTemperature&humidity=$getHumidity&ph=$getPH&rainfall=$getRainfall");
 $result=strtoupper($result);
echo "<script>alert('predicted crop is  $result') </script>";
echo "<img height='300px' width='100px' src='Crops/$result.jpg' />";


}





?>