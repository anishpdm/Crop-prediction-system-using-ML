<!DOCTYPE html>
<html lang="en">
<head>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-wEmeIV1mKuiNpC+IOBjI7aAzPcEZeedi5yW5f2yOq55WWLwNGmvvx4Um1vskeMj0" crossorigin="anonymous">

    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>php</title>
</head>
<body>
<div class="container">
<div class="row">
<div class="col col-12 col-sm-2 col-md-2 col-lg-2 col-xl-2"></div>

<div class="col col-12 col-sm-8 col-md-8 col-lg-8 col-xl-8">


<form action="" method="get">
<table class="table table-borderless">

<tr>
    <td>Name</td>
    <td><input name="fname" type="text" class="form-control"></td>
</tr>
<tr>
    <td>Roll number</td>
    <td><input name="roll" type="text" class="form-control"></td>
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
if(isset($_GET['button']))
{
 $getName=$_GET['fname'];
 $getrollnumber=$_GET['roll'];

 echo $getName;
 echo $getrollnumber;
}





?>