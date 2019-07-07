var typing_indic = '<li><div class="ticontainer"><div class="tiblock"><div class="tidot"></div><div class="tidot"></div><div class="tidot"></div></div></div></li>'

$(document).ready(function() {
  function get_response(query) {
    $('#history').append($('<li>').text(query));
    $('#history').append(typing_indic);
    $('input').prop('disabled', true);
    var mode = $('.dropdown-menu').find(".is-active").text().trim();
    // TODO: hook this endpoint up
    $.post("/chat", JSON.stringify({"message": query, "mode": mode}),  function(data) {
      $('#history li:last').remove();
      $('#history').append(
        $('<li>').html("<span class='has-text-link'>" + data['answer'] + "</span>")
      );
      $('input').val("")
      $('input').prop('disabled', false);
      $("input").focus();
    });
  }

  // keeps scroll bar at bottom of div when text
  // is being added
  function updateScroll(){
      var element = document.getElementById("chatbox");
      element.scrollTop = element.scrollHeight;
  }

  $('.input').on("keypress", function(e) {
          if (e.keyCode == 13 && e.target.value !== '') {
            get_response(e.target.value);
            updateScroll();
            return false;
          }
  });

  $('.dropdown-trigger').click(function() {
    $('.dropdown').toggleClass('is-active');
  });

  $('.dropdown-item').click(function() {
    $('.dropdown-menu').find(".is-active").removeClass("is-active");
    $(this).addClass("is-active");
  });
});
