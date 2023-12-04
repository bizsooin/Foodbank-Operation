def log_values(total_cost, grad_norms, epoch, batch_id, step, log_likelihood, 
               reinforce_loss, bl_loss, rej, length, rej_count, tb_logger, opts):
    
    avg_total_cost = total_cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Assuming `rej`, `length`, `rej_count` are tensors, calculate their mean values
    avg_rej = rej.mean().item()
    avg_length = length.mean().item()
    avg_rej_count = rej_count.float().mean().item()

    # Log values to screen
    print('epoch: {}, train_batch_id: {}, avg_total_cost: {}'.format(epoch, batch_id, avg_total_cost))
    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))
    print('avg_unfilled: {}, avg_length: {}, avg_equity: {}'.format(avg_rej, avg_length, avg_rej_count))

    # Log values to tensorboard
    if not opts.no_tensorboard:
        tb_logger.log_value('avg_total_cost', avg_total_cost, step)
        tb_logger.log_value('actor_loss', reinforce_loss.item(), step)
        tb_logger.log_value('nll', -log_likelihood.mean().item(), step)
        tb_logger.log_value('avg_unfilled', avg_rej, step)
        tb_logger.log_value('avg_length', avg_length, step)
        tb_logger.log_value('avg_equity', avg_rej_count, step)

        tb_logger.log_value('grad_norm', grad_norms[0], step)
        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)
  